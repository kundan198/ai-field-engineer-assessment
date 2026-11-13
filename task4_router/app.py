"""Task 4: LLM gateway: auth → token rate limit → failover routing → sanitised errors.

POST /v1/chat/completions   Authorization: Bearer <tenant api key>

Every error the gateway emits has exactly this shape (no upstream bodies, no
stack traces, no provider URLs/names):
    {"error": {"type": "...", "message": "...", "request_id": "req_..."}}
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

try:
    from .ratelimit import SlidingWindowTokenLimiter, tenant_id
    from .router import FallbackRouter, Provider, UpstreamError
except ImportError:
    from ratelimit import SlidingWindowTokenLimiter, tenant_id  # type: ignore
    from router import FallbackRouter, Provider, UpstreamError  # type: ignore

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llm-gateway")

DEFAULT_MAX_TOKENS = 512
ERRORS = {  # type -> (status, public message)
    "authentication_error": (401, "Missing or invalid API key."),
    "invalid_request": (400, "The request body is invalid."),
    "request_too_large": (413, "Request exceeds the per-minute token budget."),
    "rate_limit_exceeded": (429, "Token rate limit exceeded. Retry after the indicated delay."),
    "upstream_rejected": (400, "The model provider rejected the request."),
    "upstream_unavailable": (503, "All model providers are currently unavailable."),
    "internal_error": (500, "An internal error occurred."),
}


def gateway_error(kind: str, request_id: str, headers: dict[str, str] | None = None) -> JSONResponse:
    status, message = ERRORS[kind]
    return JSONResponse({"error": {"type": kind, "message": message, "request_id": request_id}},
                        status_code=status, headers={"x-request-id": request_id, **(headers or {})})


def estimate_tokens(body: dict[str, Any]) -> int:
    """Cheap, conservative pre-flight estimate (~4 chars/token) + the completion budget."""
    text = json.dumps(body.get("messages", []), ensure_ascii=False)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    return math.ceil(len(text) / 4) + int(max_tokens)


def validate(body: Any) -> bool:
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return False
    mt = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    return isinstance(mt, int) and not isinstance(mt, bool) and 0 < mt <= 32_000


def create_app(limiter: SlidingWindowTokenLimiter, router: FallbackRouter, api_keys: dict[str, str]) -> Starlette:
    """api_keys: plaintext key -> tenant name (only digests are kept in memory/DB)."""
    key_digests = {tenant_id(k): name for k, name in api_keys.items()}

    async def completions(request: Request) -> Response:
        rid = f"req_{uuid.uuid4().hex[:16]}"
        try:
            scheme, _, key = request.headers.get("authorization", "").partition(" ")
            tenant = tenant_id(key.strip()) if scheme.lower() == "bearer" and key.strip() else None
            if tenant not in key_digests:
                return gateway_error("authentication_error", rid)
            try:
                body = await request.json()
            except Exception:
                return gateway_error("invalid_request", rid)
            if not validate(body):
                return gateway_error("invalid_request", rid)
            body.pop("model", None)  # the gateway owns model selection
            body.pop("stream", None)

            est = estimate_tokens(body)
            d = await limiter.reserve(tenant, est)
            rl_headers = {"x-ratelimit-limit-tokens": str(d.limit), "x-ratelimit-remaining-tokens": str(d.remaining)}
            if d.too_large:
                return gateway_error("request_too_large", rid, rl_headers)
            if not d.allowed:
                return gateway_error("rate_limit_exceeded", rid, {**rl_headers, "retry-after": str(d.retry_after_s)})

            try:
                result = await router.complete(body, rid)
            except UpstreamError as e:
                await limiter.release(d.reservation_id)  # tenant isn't charged for our failure
                kind = "upstream_rejected" if e.kind == "client_error" else "upstream_unavailable"
                return gateway_error(kind, rid, rl_headers)
            except BaseException:
                await limiter.release(d.reservation_id)
                raise

            usage = result.body.get("usage") or {}
            actual = usage.get("total_tokens")
            if isinstance(actual, int) and actual >= 0:
                await limiter.reconcile(d.reservation_id, actual)
            # Return only the fields we promise; drop anything provider-specific.
            out = {"id": rid, "object": "chat.completion", "choices": result.body.get("choices", []),
                   "usage": usage, "fallback_used": bool(result.attempts)}
            return JSONResponse(out, headers={"x-request-id": rid, **rl_headers})
        except Exception:
            log.exception("req=%s unhandled", rid)  # full trace stays server-side
            return gateway_error("internal_error", rid)

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        await router.aclose()
        limiter.close()

    return Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"]),
                             Route("/healthz", health)], lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    env = os.environ.get
    app = create_app(
        SlidingWindowTokenLimiter(env("RATE_DB", "ratelimit.db"), int(env("TOKENS_PER_MIN", "50000"))),
        FallbackRouter(
            Provider("primary", env("PRIMARY_URL", "http://127.0.0.1:9201/primary/ok"), env("PRIMARY_MODEL", "model-a"),
                     env("PRIMARY_KEY"), float(env("PRIMARY_TIMEOUT_MS", "3000")) / 1000),
            Provider("secondary", env("SECONDARY_URL", "http://127.0.0.1:9201/secondary/ok"),
                     env("SECONDARY_MODEL", "model-b"), env("SECONDARY_KEY"),
                     float(env("SECONDARY_TIMEOUT_MS", "10000")) / 1000),
        ),
        json.loads(env("TENANT_KEYS", '{"sk-tenant-a": "tenant-a", "sk-tenant-b": "tenant-b"}')),
    )
    uvicorn.run(app, host="127.0.0.1", port=int(env("PORT", "9200")), log_config=None)
