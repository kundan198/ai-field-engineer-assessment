import asyncio
import json
import threading
import time

import httpx
import pytest

from task4_router import mock_providers
from task4_router.app import create_app
from task4_router.ratelimit import SlidingWindowTokenLimiter
from task4_router.router import FallbackRouter, Provider


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- limiter
@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def limiter(tmp_path, clock):
    lim = SlidingWindowTokenLimiter(str(tmp_path / "rl.db"), 50_000, 60, clock=clock)
    yield lim
    lim.close()


def test_admits_up_to_limit_then_denies(limiter):
    assert limiter.reserve_sync("t", 30_000).allowed
    assert limiter.reserve_sync("t", 20_000).allowed
    d = limiter.reserve_sync("t", 1)
    assert not d.allowed and d.remaining == 0


def test_tenants_isolated(limiter):
    assert limiter.reserve_sync("a", 50_000).allowed
    assert limiter.reserve_sync("b", 50_000).allowed


def test_sliding_window_eviction_and_exact_retry_after(limiter, clock):
    limiter.reserve_sync("t", 20_000)          # t=0
    clock.t += 30
    limiter.reserve_sync("t", 30_000)          # t=30
    clock.t += 10                              # t=40: window full
    d = limiter.reserve_sync("t", 10_000)
    assert not d.allowed and d.retry_after_s == 20   # the t=0 row expires at t=60
    clock.t += 20.001                          # t=60.001: first row slides out
    assert limiter.reserve_sync("t", 10_000).allowed
    assert limiter.usage_sync("t") == 40_000
    rows = limiter._db.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
    assert rows == 2                           # expired row physically deleted


def test_boundary_is_exclusive(limiter, clock):
    limiter.reserve_sync("t", 50_000)
    clock.t += 60                              # exactly window later -> expired
    assert limiter.reserve_sync("t", 50_000).allowed


def test_reconcile_and_release(limiter):
    d = limiter.reserve_sync("t", 40_000)       # pessimistic estimate
    limiter.reconcile_sync(d.reservation_id, 1_000)
    assert limiter.reserve_sync("t", 49_000).allowed
    d2 = limiter.reserve_sync("t", 0)
    limiter.release_sync(d2.reservation_id)
    assert limiter.usage_sync("t") == 50_000


def test_single_request_larger_than_budget(limiter):
    d = limiter.reserve_sync("t", 50_001)
    assert d.too_large and not d.allowed


def test_persists_on_disk_across_instances(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    a = SlidingWindowTokenLimiter(path, clock=clock)
    a.reserve_sync("t", 45_000)
    a.close()
    b = SlidingWindowTokenLimiter(path, clock=clock)
    assert not b.reserve_sync("t", 10_000).allowed
    b.close()


async def test_concurrent_reservations_never_overshoot(limiter):
    results = await asyncio.gather(*[limiter.reserve("t", 5_000) for _ in range(40)])
    assert sum(r.allowed for r in results) == 10
    assert limiter.usage_sync("t") == 50_000


def test_two_processes_style_connections_share_budget(tmp_path, clock):
    """Two limiter instances (= two gateway workers) on the same file, hammered from threads."""
    path = str(tmp_path / "rl.db")
    a, b = SlidingWindowTokenLimiter(path, clock=clock), SlidingWindowTokenLimiter(path, clock=clock)
    ok = []
    def worker(lim):
        for _ in range(20):
            ok.append(lim.reserve_sync("t", 2_000).allowed)
    ts = [threading.Thread(target=worker, args=(x,)) for x in (a, b, a, b)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(ok) == 25
    a.close(); b.close()


# ---------------------------------------------------------------- router + app
KEY = "sk-tenant-a"


def make(tmp_path, primary="ok", secondary="ok", timeout=0.3, limit=50_000):
    mock_providers.HITS.clear()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_providers.app), base_url="http://p")
    router = FallbackRouter(Provider("primary", f"http://p/primary/{primary}", "m-a", timeout_s=timeout),
                            Provider("secondary", f"http://p/secondary/{secondary}", "m-b", timeout_s=timeout),
                            client=client)
    limiter = SlidingWindowTokenLimiter(str(tmp_path / "rl.db"), limit)
    app = create_app(limiter, router, {KEY: "tenant-a"})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw"), limiter


BODY = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}


async def post(c, body=BODY, key=KEY):
    return await c.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"} if key else {})


def assert_sanitised(r, kind):
    j = r.json()
    assert set(j) == {"error"} and set(j["error"]) == {"type", "message", "request_id"}
    assert j["error"]["type"] == kind
    blob = r.text.lower()
    for leak in ("traceback", "internal", "gpu", "shard", "primary", "secondary", "org-", "http://"):
        assert leak not in blob, leak


def test_default_primary_timeout_is_3000ms():
    assert Provider("p", "u", "m").timeout_s == 3.0


async def test_primary_success(tmp_path):
    c, lim = make(tmp_path)
    r = await post(c)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello from primary"
    assert r.json()["fallback_used"] is False and "internal_node" not in r.text
    assert mock_providers.HITS == ["primary/ok"]
    assert lim.usage_sync(__import__("task4_router.ratelimit", fromlist=["x"]).tenant_id(KEY)) == 15  # reconciled


@pytest.mark.parametrize("bad", ["429", "500", "garbage"])
async def test_failover_on_primary_failure(tmp_path, bad):
    c, _ = make(tmp_path, primary=bad)
    r = await post(c)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello from secondary"
    assert r.json()["fallback_used"] is True
    assert mock_providers.HITS == [f"primary/{bad}", "secondary/ok"]


async def test_failover_on_timeout_is_prompt(tmp_path):
    c, _ = make(tmp_path, primary="slow", timeout=0.3)
    t0 = time.perf_counter()
    r = await post(c)
    elapsed = time.perf_counter() - t0
    assert r.json()["choices"][0]["message"]["content"] == "hello from secondary"
    assert 0.3 <= elapsed < 1.5   # waited for the deadline, not the 10 s slow upstream


async def test_no_failover_on_client_4xx(tmp_path):
    c, _ = make(tmp_path, primary="400")
    r = await post(c)
    assert r.status_code == 400
    assert_sanitised(r, "upstream_rejected")
    assert mock_providers.HITS == ["primary/400"]


async def test_both_down_is_sanitised_and_refunded(tmp_path):
    from task4_router.ratelimit import tenant_id
    c, lim = make(tmp_path, primary="500", secondary="500")
    r = await post(c)
    assert r.status_code == 503
    assert_sanitised(r, "upstream_unavailable")
    assert lim.usage_sync(tenant_id(KEY)) == 0


async def test_both_timeout(tmp_path):
    c, _ = make(tmp_path, primary="slow", secondary="slow", timeout=0.2)
    r = await post(c)
    assert r.status_code == 503
    assert_sanitised(r, "upstream_unavailable")


async def test_rate_limited_returns_429_with_retry_after(tmp_path):
    c, _ = make(tmp_path, primary="slow", timeout=0.2, limit=1_000)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 600}
    first = asyncio.create_task(post(c, body))       # holds a 600+ token reservation while in flight
    await asyncio.sleep(0.05)
    r = await post(c, body)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert_sanitised(r, "rate_limit_exceeded")
    await first


async def test_auth_and_validation_errors(tmp_path):
    c, _ = make(tmp_path)
    assert_sanitised(await post(c, key=None), "authentication_error")
    assert_sanitised(await post(c, key="sk-wrong"), "authentication_error")
    assert_sanitised(await post(c, body={"messages": []}), "invalid_request")
    assert_sanitised(await post(c, body={"messages": [{}], "max_tokens": True}), "invalid_request")
    r = await c.post("/v1/chat/completions", content=b"{nope", headers={"Authorization": f"Bearer {KEY}"})
    assert_sanitised(r, "invalid_request")


async def test_request_bigger_than_budget(tmp_path):
    c, _ = make(tmp_path, limit=1_000)
    r = await post(c, {"messages": [{"role": "user", "content": "x"}], "max_tokens": 5_000})
    assert r.status_code == 413
