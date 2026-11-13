"""Primary → secondary model failover with hard per-attempt deadlines.

Failover triggers (required): HTTP 429, or no complete response within 3000 ms.
Also fails over on 5xx / connection errors / unparseable bodies (same outcome
for the caller: the primary cannot serve). Does NOT fail over on other 4xx, since that
is the caller's fault and the backup would reject it too.

Timeout semantics: `asyncio.timeout()` wraps the *whole* attempt (connect, send,
status, body read). When it fires the in-flight httpx request is cancelled and
its connection discarded, so a late primary reply can never be returned, double-
counted, or race with the secondary's answer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("llm-router")

FAILOVER_KINDS = {"rate_limited", "timeout", "server_error", "network", "bad_response"}


@dataclass
class Provider:
    name: str
    url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 3.0


class UpstreamError(Exception):
    """Internal only; never serialised to clients."""

    def __init__(self, provider: str, kind: str, status: int | None = None, detail: str = ""):
        super().__init__(f"{provider}:{kind}:{status}")
        self.provider, self.kind, self.status, self.detail = provider, kind, status, detail


@dataclass
class CircuitBreaker:
    """Skip a provider that keeps failing, so we don't pay its 3 s timeout on every request."""
    threshold: int = 5
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def available(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return True  # half-open: let one attempt through
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self.failures, self.opened_at = 0, None
        else:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.monotonic()


@dataclass
class RouteResult:
    provider: str
    body: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)


class FallbackRouter:
    def __init__(self, primary: Provider, secondary: Provider, client: httpx.AsyncClient | None = None):
        self.primary, self.secondary = primary, secondary
        self.http = client or httpx.AsyncClient(limits=httpx.Limits(max_connections=200))
        self.breaker = CircuitBreaker()

    async def _attempt(self, p: Provider, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if p.api_key:
            headers["authorization"] = f"Bearer {p.api_key}"
        try:
            async with asyncio.timeout(p.timeout_s):
                r = await self.http.post(p.url, json={**payload, "model": p.model}, headers=headers)
                raw = await r.aread()
        except TimeoutError:
            raise UpstreamError(p.name, "timeout", detail=f">{p.timeout_s * 1000:.0f}ms") from None
        except httpx.TimeoutException:
            raise UpstreamError(p.name, "timeout") from None
        except httpx.HTTPError as e:
            raise UpstreamError(p.name, "network", detail=type(e).__name__) from None
        if r.status_code == 429:
            raise UpstreamError(p.name, "rate_limited", 429)
        if r.status_code >= 500:
            raise UpstreamError(p.name, "server_error", r.status_code, raw[:200].decode(errors="replace"))
        if r.status_code >= 400:
            raise UpstreamError(p.name, "client_error", r.status_code, raw[:200].decode(errors="replace"))
        try:
            body = r.json()
            assert isinstance(body, dict) and isinstance(body.get("choices"), list)
        except Exception:
            raise UpstreamError(p.name, "bad_response", r.status_code) from None
        return body

    async def complete(self, payload: dict[str, Any], request_id: str) -> RouteResult:
        attempts: list[dict[str, Any]] = []
        if self.breaker.available():
            t0 = time.perf_counter()
            try:
                body = await self._attempt(self.primary, payload)
                self.breaker.record(True)
                return RouteResult(self.primary.name, body, attempts)
            except UpstreamError as e:
                attempts.append({"provider": e.provider, "kind": e.kind, "status": e.status,
                                 "ms": round((time.perf_counter() - t0) * 1000)})
                # Internal log keeps the detail; the client never sees it.
                log.warning("req=%s primary failed kind=%s status=%s detail=%r", request_id, e.kind, e.status, e.detail)
                if e.kind not in FAILOVER_KINDS:
                    raise
                self.breaker.record(False)
        else:
            attempts.append({"provider": self.primary.name, "kind": "circuit_open"})

        t0 = time.perf_counter()
        try:
            body = await self._attempt(self.secondary, payload)
        except UpstreamError as e:
            log.error("req=%s secondary failed kind=%s status=%s detail=%r", request_id, e.kind, e.status, e.detail)
            raise
        log.info("req=%s served by secondary after %s", request_id, attempts)
        return RouteResult(self.secondary.name, body, attempts)

    async def aclose(self) -> None:
        await self.http.aclose()
