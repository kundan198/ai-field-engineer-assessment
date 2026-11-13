import json
import random
import time

import httpx
import pytest

from task3_pii_stream import mock_llm
from task3_pii_stream.gateway import create_app
from task3_pii_stream.redactor import MAX_PII_LEN, StreamRedactor

TEXT = ("Hi, mail john.doe@example.com or call. SSN 123-45-6789, alt 078 05 1120, card 4111 1111 1111 1111 "
        "and amex 378282246310005 and fake 1234 5678 9012 3456. Bye now")
EXPECTED = ("Hi, mail [REDACTED] or call. SSN [REDACTED], alt [REDACTED], card [REDACTED] "
            "and amex [REDACTED] and fake 1234 5678 9012 3456. Bye now")


def run(text, sizes):
    r, out, i, peak = StreamRedactor(), [], 0, 0
    while i < len(text):
        n = sizes()
        out.append(r.feed(text[i:i + n]))
        peak = max(peak, r.held)
        i += n
    out.append(r.flush())
    return "".join(out), peak


@pytest.mark.parametrize("seed", range(300))
def test_random_chunking_never_leaks(seed):
    rnd = random.Random(seed)
    out, _ = run(TEXT, lambda: rnd.randint(1, 12))
    assert out == EXPECTED


def test_char_by_char():
    assert run(TEXT, lambda: 1)[0] == EXPECTED


def test_prose_is_released_immediately():
    r = StreamRedactor()
    assert r.feed("Hello there, how are ") == "Hello there, how are "
    assert r.feed("you") == ""          # last word held: could be start of an email
    assert r.feed(" today") == "you "   # released once a boundary arrives


def test_memory_bounded_on_adversarial_input():
    out, peak = run("x" * 20000 + " end", lambda: 50)
    assert peak <= MAX_PII_LEN + 50
    assert out == "x" * 20000 + " end"


def test_email_with_unbounded_prefix_still_redacted():
    # 500-char token followed by an email: only the (<=64 char) local part matches.
    text = "a" * 500 + "@corp.com done"
    out, _ = run(text, lambda: 7)
    assert "@corp.com" not in out and out.endswith("[REDACTED] done")


def test_no_false_positive_on_non_luhn_or_short_numbers():
    text = "Order 1234 5678 9012 3456, ticket 12345, phone-ish 555-1234, year 2026."
    assert run(text, lambda: 3)[0] == text


@pytest.fixture
async def gw():
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm.app))
    app = create_app("http://llm/v1/chat/completions", client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c
    await upstream.aclose()


async def test_gateway_streams_redacted_sse(gw):
    body = {"model": "m", "stream": True, "mock_delay": 0, "messages": []}
    events, text = [], ""
    async with gw.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                events.append(line[5:].strip())
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    for c in chunks:
        for ch in c["choices"]:
            text += ch["delta"].get("content") or ""
    assert "078-05-1120" not in text and "4111" not in text and "acme-corp" not in text
    assert text.count("[REDACTED]") == 3
    assert "Order 1234 5678 9012 3456 is not a card" in text
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(ch.get("finish_reason") == "stop" for c in chunks for ch in c["choices"])
    assert len(chunks) > 10  # still incremental, not one big blob


def _serve(app):
    import socket, threading, uvicorn
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.01)
    return srv, port


async def test_gateway_ttft_not_blocked_by_full_response():
    """Over real sockets, with 20ms between upstream chunks, the first content byte must reach
    the client long before the stream ends (i.e. the gateway is not buffering the response)."""
    llm, llm_port = _serve(mock_llm.app)
    gw_srv, gw_port = _serve(create_app(f"http://127.0.0.1:{llm_port}/v1/chat/completions"))
    try:
        body = {"model": "m", "stream": True, "mock_delay": 0.02, "messages": []}
        first = None
        async with httpx.AsyncClient() as c:
            t0 = time.perf_counter()
            async with c.stream("POST", f"http://127.0.0.1:{gw_port}/v1/chat/completions", json=body) as r:
                async for line in r.aiter_lines():
                    if first is None and line.startswith("data:") and '"content":"' in line and '"content":""' not in line:
                        first = time.perf_counter() - t0
            total = time.perf_counter() - t0
        assert first is not None and first < total / 5, (first, total)
    finally:
        gw_srv.should_exit = llm.should_exit = True


async def test_gateway_non_streaming(gw):
    r = await gw.post("/v1/chat/completions", json={"model": "m", "messages": []})
    content = r.json()["choices"][0]["message"]["content"]
    assert content.count("[REDACTED]") == 3
