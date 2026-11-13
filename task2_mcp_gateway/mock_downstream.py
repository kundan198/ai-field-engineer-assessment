"""A tiny JSON-RPC-over-HTTP mock MCP server used behind the gateway."""
from __future__ import annotations

import json
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

TOOLS = [
    {"name": "get_customer_record", "description": "Read a customer", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object"}},
]

CALLS: list[dict] = []  # observed requests (for tests)


async def mcp(request: Request) -> JSONResponse:
    msg = json.loads(await request.body())
    CALLS.append({"msg": msg, "headers": dict(request.headers)})
    m, id_ = msg.get("method"), msg.get("id")
    if m == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "mock", "version": "0"}}
    elif m == "tools/list":
        result = {"tools": TOOLS}
    elif m == "tools/call":
        result = {"content": [{"type": "text", "text": f"ran {msg['params']['name']}"}]}
    else:
        return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": "Method not found"}})
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "result": result})


app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9001")))
