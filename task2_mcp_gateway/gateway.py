"""Task 2: MCP Security Gateway (HTTP JSON-RPC reverse proxy).

    AI agent ──HTTP POST /mcp (Bearer JWT)──▶ gateway ──▶ downstream MCP server

Pipeline per request
  1. AuthN   : Bearer token must be a valid HS256 JWT; role taken from `role` claim.
               Missing/invalid -> HTTP 401 + JSON-RPC error -32001 (no forwarding).
  2. Parse   : strict JSON (duplicate keys rejected to kill parser-differential
               smuggling), JSON-RPC 2.0 envelope checks (-32700 / -32600).
  3. AuthZ   : tools/list -> forwarded transparently.
               tools/call -> params.name must be a string; names starting with
               `admin_` (case-insensitive, whitespace-trimmed) require role=admin,
               otherwise the gateway answers -32001 "Unauthorized Tool Call"
               itself and the downstream server is never contacted.
  4. Forward : the *canonical re-serialisation* we authorised is what gets sent
               (not the raw bytes), the caller's token is stripped (no token
               passthrough), identity is passed as X-Gateway-* headers, and the
               downstream response (JSON or SSE) is streamed back.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

try:
    from .tokens import TokenError, verify
except ImportError:  # run as a script
    from tokens import TokenError, verify  # type: ignore

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mcp-gateway")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32001

VALID_ROLES = {"admin", "viewer"}
ADMIN_PREFIX = "admin_"
MAX_BODY = 1 * 1024 * 1024
# Request headers we forward downstream (allowlist; Authorization/Cookie are never forwarded).
FWD_REQ_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FWD_RESP_HEADERS = {"content-type", "mcp-session-id", "cache-control"}


def rpc_error(id_: Any, code: int, message: str, *, status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}},
                        status_code=status, headers=headers)


class DuplicateKeyError(ValueError):
    pass


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise DuplicateKeyError(k)
        d[k] = v
    return d


def requires_admin(tool_name: str) -> bool:
    return tool_name.strip().lower().startswith(ADMIN_PREFIX)


def authorize(msg: dict[str, Any], role: str) -> tuple[int, str] | None:
    """Return (code, message) to deny, or None to allow."""
    method = msg["method"]
    if method == "tools/call":
        params = msg.get("params")
        if not isinstance(params, dict) or not isinstance(params.get("name"), str) or not params["name"].strip():
            return INVALID_PARAMS, "Invalid params: tools/call requires string params.name"
        if requires_admin(params["name"]) and role != "admin":
            return UNAUTHORIZED, "Unauthorized Tool Call"
    # tools/list and every other MCP method: forwarded transparently.
    return None


def create_app(downstream_url: str, jwt_secret: str, client: httpx.AsyncClient | None = None) -> Starlette:
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def mcp_endpoint(request: Request) -> Response:
        # ---- 1. AuthN -------------------------------------------------------
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return rpc_error(None, UNAUTHORIZED, "Missing bearer token", status=401,
                             headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'})
        try:
            claims = verify(token.strip(), jwt_secret)
        except TokenError as e:
            log.info("auth rejected: %s", e)
            return rpc_error(None, UNAUTHORIZED, "Invalid bearer token", status=401,
                             headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})
        role = claims.get("role")
        if role not in VALID_ROLES:
            return rpc_error(None, UNAUTHORIZED, "Token has no valid role", status=403)
        subject = str(claims.get("sub", "unknown"))

        # ---- 2. Parse -------------------------------------------------------
        body = await request.body()
        if len(body) > MAX_BODY:
            return rpc_error(None, INVALID_REQUEST, "Request too large", status=413)
        try:
            msg = json.loads(body, object_pairs_hook=_no_dupes)
        except DuplicateKeyError:
            return rpc_error(None, INVALID_REQUEST, "Invalid Request: duplicate JSON keys")
        except (ValueError, UnicodeDecodeError):
            return rpc_error(None, PARSE_ERROR, "Parse error")
        if isinstance(msg, list):
            # JSON-RPC batching was removed from MCP (2025-06-18); refusing it also
            # closes the "hide an admin_ call inside a batch" bypass.
            return rpc_error(None, INVALID_REQUEST, "Invalid Request: batching is not supported")
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return rpc_error(None, INVALID_REQUEST, "Invalid Request")
        id_ = msg.get("id")
        if not isinstance(id_, (str, int, type(None))) or isinstance(id_, bool):
            return rpc_error(None, INVALID_REQUEST, "Invalid Request: bad id")
        if "method" not in msg:
            # A JSON-RPC *response* from the client (e.g. answering a sampling request): pass through.
            pass
        elif not isinstance(msg["method"], str):
            return rpc_error(id_, INVALID_REQUEST, "Invalid Request: method must be a string")

        # ---- 3. AuthZ -------------------------------------------------------
        if "method" in msg:
            denial = authorize(msg, role)
            if denial:
                code, message = denial
                log.warning("DENY sub=%s role=%s method=%s tool=%r", subject, role, msg["method"],
                            (msg.get("params") or {}).get("name") if isinstance(msg.get("params"), dict) else None)
                return rpc_error(id_, code, message)
            log.info("ALLOW sub=%s role=%s method=%s", subject, role, msg["method"])

        # ---- 4. Forward -----------------------------------------------------
        headers = {k: v for k, v in request.headers.items() if k.lower() in FWD_REQ_HEADERS}
        headers.setdefault("content-type", "application/json")
        headers.setdefault("accept", "application/json, text/event-stream")
        headers["x-gateway-user"] = subject
        headers["x-gateway-role"] = role
        canonical = json.dumps(msg, separators=(",", ":")).encode()
        try:
            upstream_req = http.build_request("POST", downstream_url, content=canonical, headers=headers)
            upstream = await http.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            log.error("downstream unreachable: %s", type(e).__name__)
            return rpc_error(id_, INTERNAL_ERROR, "Downstream MCP server unavailable", status=502)

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in FWD_RESP_HEADERS}
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                 headers=resp_headers, background=BackgroundTask(upstream.aclose))

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        if client is None:
            await http.aclose()

    return Starlette(routes=[Route("/mcp", mcp_endpoint, methods=["POST"]), Route("/healthz", health)],
                     lifespan=lifespan)


app = None
if __name__ == "__main__":
    import uvicorn

    app = create_app(os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp"),
                     os.environ.get("GATEWAY_JWT_SECRET", "dev-secret"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9000")), log_config=None)
