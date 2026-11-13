"""Task 3: LLM gateway endpoint with real-time streaming PII redaction.

POST /v1/chat/completions  (OpenAI-compatible; works with any OpenAI-style upstream)

Stream path:
  client ◀─SSE─ gateway ◀─SSE─ provider
  * Upstream SSE is consumed line-by-line (httpx.aiter_lines) and never buffered whole.
  * Each `choices[i].delta.content` goes through a per-choice StreamRedactor; the
    chunk is re-emitted immediately with the redacted (possibly shorter) content.
    Chunks with nothing safe to emit yet are dropped, except role/finish/tool events
    which always pass through, so the client sees the first token as soon as it is safe.
  * Before a choice's finish_reason (or [DONE] / upstream EOF) the held tail is flushed.
  * Pull-based async generator: if the client is slow we stop reading upstream
    (natural backpressure); if the client disconnects the generator is closed and the
    upstream connection is released via `async with`.
Memory per stream: O(MAX_PII_LEN × n_choices), independent of response length.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

try:
    from .redactor import StreamRedactor
except ImportError:
    from redactor import StreamRedactor  # type: ignore

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llm-gateway")


def _sse(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


def _flush_chunk(template: dict, index: int, text: str) -> dict:
    return {**{k: v for k, v in template.items() if k != "choices"},
            "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}]}


async def redact_sse(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    redactors: dict[int, StreamRedactor] = {}
    template: dict = {}
    async for line in lines:
        if not line.startswith("data:"):
            continue  # ignore comments/keepalives/event: lines
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        template = chunk
        out_choices = []
        for ch in chunk.get("choices", []):
            idx = ch.get("index", 0)
            r = redactors.setdefault(idx, StreamRedactor())
            delta = dict(ch.get("delta") or {})
            content = delta.get("content")
            if isinstance(content, str):
                delta["content"] = r.feed(content)
            if ch.get("finish_reason") is not None:
                delta["content"] = (delta.get("content") or "") + r.flush()
            meaningful = delta.get("content") or set(delta) - {"content"} or ch.get("finish_reason") is not None
            if meaningful:
                out_choices.append({**ch, "delta": delta})
        if out_choices or not chunk.get("choices"):  # usage-only chunks pass through
            yield _sse({**chunk, "choices": out_choices})
    # Upstream ended ([DONE] or EOF): release anything still held.
    for idx, r in redactors.items():
        tail = r.flush()
        if tail:
            yield _sse(_flush_chunk(template, idx, tail))
    yield b"data: [DONE]\n\n"


def create_app(upstream_url: str, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> Starlette:
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def completions(request: Request) -> Response:
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:
            return JSONResponse({"error": {"type": "invalid_request", "message": "Body must be a JSON object"}}, 400)
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        if not body.get("stream"):
            try:
                r = await http.post(upstream_url, json=body, headers=headers)
            except httpx.HTTPError:
                return JSONResponse({"error": {"type": "upstream_unavailable", "message": "Upstream unavailable"}}, 502)
            data = r.json()
            for ch in data.get("choices", []):
                msg = ch.get("message") or {}
                if isinstance(msg.get("content"), str):
                    rd = StreamRedactor()
                    msg["content"] = rd.feed(msg["content"]) + rd.flush()
            return JSONResponse(data, r.status_code)

        req = http.build_request("POST", upstream_url, json=body, headers=headers)
        try:
            upstream = await http.send(req, stream=True)
        except httpx.HTTPError:
            return JSONResponse({"error": {"type": "upstream_unavailable", "message": "Upstream unavailable"}}, 502)
        if upstream.status_code != 200:
            await upstream.aclose()
            return JSONResponse({"error": {"type": "upstream_error", "message": "Upstream returned an error"}},
                                502 if upstream.status_code >= 500 else upstream.status_code)

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for b in redact_sse(upstream.aiter_lines()):
                    yield b
            finally:
                await upstream.aclose()  # also runs on client disconnect

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    return Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9101/v1/chat/completions"),
                           os.environ.get("UPSTREAM_API_KEY")),
                host="127.0.0.1", port=int(os.environ.get("PORT", "9100")), log_config=None)
