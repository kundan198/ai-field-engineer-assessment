"""Minimal HS256 JWT (stdlib only) used to carry the caller's role.

The gateway trusts *only* a signed, unexpired token, never a role sent in a
plain header or in the JSON-RPC body.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class TokenError(Exception):
    pass


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def mint(claims: dict[str, Any], secret: str, ttl_s: int = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    body = {"iat": int(time.time()), "exp": int(time.time()) + ttl_s, **claims}
    signing_input = f"{_b64e(json.dumps(header).encode())}.{_b64e(json.dumps(body).encode())}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64e(sig)}"


def verify(token: str, secret: str, *, leeway_s: int = 30) -> dict[str, Any]:
    try:
        h64, p64, s64 = token.split(".")
        header = json.loads(_b64d(h64))
        claims = json.loads(_b64d(p64))
        sig = _b64d(s64)
    except Exception as e:
        raise TokenError("malformed token") from e
    if header.get("alg") != "HS256":  # blocks alg=none / alg confusion
        raise TokenError("unsupported alg")
    expected = hmac.new(secret.encode(), f"{h64}.{p64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise TokenError("bad signature")
    if not isinstance(claims, dict):
        raise TokenError("bad claims")
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)) or exp + leeway_s < time.time():
        raise TokenError("expired")
    return claims


if __name__ == "__main__":  # python tokens.py <sub> <role>  -> prints a token
    import os
    import sys

    print(mint({"sub": sys.argv[1], "role": sys.argv[2]}, os.environ.get("GATEWAY_JWT_SECRET", "dev-secret")))
