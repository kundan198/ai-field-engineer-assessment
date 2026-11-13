"""Mock providers: /{name}/{behaviour} where behaviour ∈ ok | 429 | slow | 500 | 400 | garbage."""
from __future__ import annotations

import asyncio
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

HITS: list[str] = []
SLOW_S = 10.0


async def provider(request: Request):
    name, behaviour = request.path_params["name"], request.path_params["behaviour"]
    HITS.append(f"{name}/{behaviour}")
    body = await request.json()
    if behaviour == "429":
        return JSONResponse({"error": {"message": "quota exhausted for org-internal-123"}}, 429)
    if behaviour == "500":
        return PlainTextResponse('Traceback (most recent call last):\n  File "/srv/internal/llm.py", line 42\n'
                                 "RuntimeError: GPU node gpu-17.internal OOM", 500)
    if behaviour == "400":
        return JSONResponse({"error": {"message": "context length exceeded at shard x9"}}, 400)
    if behaviour == "garbage":
        return PlainTextResponse("<html>bad gateway</html>", 200)
    if behaviour == "slow":
        await asyncio.sleep(SLOW_S)
    return JSONResponse({"id": "x", "model": body.get("model"), "internal_node": "gpu-3",
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hello from {name}"},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


app = Starlette(routes=[Route("/{name}/{behaviour}", provider, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9201")))
