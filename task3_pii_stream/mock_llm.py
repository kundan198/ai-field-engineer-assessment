"""Mock OpenAI-style streaming provider that deliberately splits PII across chunks."""
from __future__ import annotations

import asyncio
import json
import os
import random

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_TEXT = ("Sure! The customer's email is jane.roe+billing@acme-corp.io and her SSN is 078-05-1120. "
                "The card on file is 4111-1111-1111-1111 (expires 09/29). Order 1234 5678 9012 3456 is not a card. "
                "Let me know if you need anything else.")


async def chat(request: Request):
    body = await request.json()
    text = body.get("mock_text", DEFAULT_TEXT)
    delay = float(body.get("mock_delay", 0.01))
    seed = body.get("mock_seed", 1)
    if not body.get("stream"):
        return JSONResponse({"id": "c1", "object": "chat.completion",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}]})

    async def gen():
        rnd = random.Random(seed)
        base = {"id": "c1", "object": "chat.completion.chunk", "model": "mock"}
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        i = 0
        while i < len(text):
            n = rnd.randint(1, 6)
            piece, i = text[i:i + n], i + n
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(delay)
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9101")))
