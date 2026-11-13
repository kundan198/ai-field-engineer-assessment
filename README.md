# AI Field Developer / Solutions Engineer: Assessment Solutions

Python 3.12 · official `mcp` SDK (2.3) · Pydantic v2 · Starlette + httpx · stdlib `sqlite3`

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q          # 369 tests, ~3 s
```

| Task | Code | Tests |
|---|---|---|
| 1 · MCP server (stdio, strict validation) | `task1_mcp_server/server.py` | `tests/test_task1_stdio.py` |
| 2 · MCP security gateway | `task2_mcp_gateway/` | `tests/test_task2_gateway.py` |
| 3 · Streaming PII redaction | `task3_pii_stream/` | `tests/test_task3_pii.py` |
| 4 · Rate limiter + failover router | `task4_router/` | `tests/test_task4_router.py` |

---

## Task 1: Custom MCP server

Run it: `.venv/bin/python task1_mcp_server/server.py` (stdio). Inspect it with `npx @modelcontextprotocol/inspector .venv/bin/python task1_mcp_server/server.py`.

**Keeping stdout clean**
- Nothing calls `print()`. All logging uses `logging` → `sys.stderr`.
- Backup protection: while serving, the SDK's `stdio_server()` points fd 1 at stderr, so a stray print from a dependency can't corrupt the protocol stream.
- *Verified*, not assumed: the test runs the server as a real subprocess over raw pipes and `json.loads` **every** stdout line. It also checks that the log lines landed on stderr.

**Validation (Pydantic, `strict=True`, `extra="forbid"`)**
- `customer_id`: `^CUST-[0-9]{5}$`. "XXXXX" is read as 5 digits.
- `amount`: a real float/int, not a string or a bool (strict mode rejects `"10"` and `true`). It must be greater than 0 and at most 10,000, finite (no NaN/inf), and have at most 2 decimal places.
- `reason`: at least 10 characters *after trimming whitespace*, at most 500.
- Unknown keys are rejected. The advertised `inputSchema` is generated from the same models (`additionalProperties: false`), so the schema and the enforcement can't drift apart.

**Error mapping**

| Situation | Response |
|---|---|
| Bad arguments | JSON-RPC `-32602` with `data: [{field, message, type}]`. Every bad field is listed, and the raw input is never echoed back. |
| Unknown tool | `-32602` |
| Unexpected exception | `-32603 "Internal error"`. The traceback goes to stderr only. |
| Business rule (customer not found, refund > refundable balance) | `CallToolResult(isError=true)`. The MCP spec says tool-execution failures belong in the result so the model can self-correct. |

> Note: newer MCP spec revisions recommend reporting *input-validation* failures as `isError` results too. I used `-32602` because the brief explicitly asks for JSON-RPC error codes; switching is a one-line change in `handle_call_tool`.

## Task 2: MCP security gateway

```bash
.venv/bin/python task2_mcp_gateway/mock_downstream.py                 # :9001
.venv/bin/python task2_mcp_gateway/gateway.py                         # :9000 → :9001
TOKEN=$(.venv/bin/python task2_mcp_gateway/tokens.py bob viewer)
curl -s localhost:9000/mcp -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
# → {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
```

Each request goes through four steps: **authenticate → parse → authorize → forward**.
- **Role comes from a signed token.** The role is read from an HS256 JWT, checked for signature and expiry, and `alg` is pinned so `alg:none` is rejected. A missing or bad token gets **401** plus `-32001`. A role other than admin/viewer gets **403**.
- **Authorization rule.** `tools/list` is forwarded transparently. For `tools/call`, `params.name` must be a non-empty string. If it starts with `admin_` (case-insensitive, after trimming), the role must be `admin`. Otherwise the gateway itself answers `-32001 Unauthorized Tool Call`, echoing the original `id`, and **the downstream server is never called**. The tests assert this.
- **Bypass attempts closed off:**
  - JSON **batches** are rejected (`-32600`). MCP dropped batching in 2025-06-18, and a batch could otherwise hide an `admin_` call.
  - **Duplicate JSON keys** are rejected. `{"name":"admin_x","name":"safe"}` would otherwise be read differently by the gateway and the downstream parser.
  - The gateway forwards its **own re-serialized copy** of the message it authorized, not the raw bytes.
- **No token passthrough.** `Authorization` and cookies are stripped. Identity is passed on as `X-Gateway-User` / `X-Gateway-Role`. Request and response headers are forwarded only from an allowlist.
- **Responses are streamed back** (`aiter_raw`), so Streamable-HTTP SSE responses work. If the downstream is unreachable, the client gets 502 `-32603` with a generic message.

## Task 3: Streaming PII redaction

```bash
.venv/bin/python task3_pii_stream/mock_llm.py      # :9101, splits PII across chunks
.venv/bin/python task3_pii_stream/gateway.py       # :9100
curl -N localhost:9100/v1/chat/completions -d '{"stream":true,"messages":[]}'
```

**The problem:** a provider can send `"jo" | "hn.doe@ex" | "ample.com"`. A regex run on each chunk separately leaks the email, and buffering the whole response destroys TTFT.

**How `StreamRedactor` works** (`redactor.py`): each delta is appended to a small buffer, and the redactor computes a **safe cut point**, the furthest position that no future match can reach back past. Everything before the cut is redacted and emitted immediately; the rest is held. The cut is the larger of:
1. **The start of the trailing PII-shaped run.** PII never spans whitespace, except a single space or dash *between digits* (cards, SSNs). So `"Hello there, how are "` goes out the instant the space arrives, and only the last partial word is ever held.
2. **`len(buf) − MAX_PII_LEN`.** Every pattern is length-bounded, so a match can't start further back than this. This puts a hard cap on memory even for a 20,000-character unbroken token. The tests cover this.

The cut is then pulled back to the start of any complete match that crosses it. One character of already-emitted text is kept as left context so look-behinds behave exactly as they would on the full text.

**Patterns:** email; SSN in `###-##-####` or `### ## ####` form (invalid 000/666/9xx areas excluded); and credit cards of 13–19 digits with optional separators, **Luhn-validated** so order numbers aren't redacted.

**Gateway:** parses SSE line by line, keeps one redactor per `choice.index`, re-emits each chunk at once with the redacted content, and always passes role/finish/tool deltas through. The held tail is flushed before `finish_reason` and before `[DONE]`/EOF. Because the gateway is a pull-based generator, a slow client naturally slows upstream reads (backpressure), and a client disconnect closes the upstream connection. Memory is O(MAX_PII_LEN × choices), independent of response length.

**Tests:**
- 300 random chunkings plus char-by-char, all producing the exact expected output.
- An adversarial memory test.
- A false-positive test.
- An end-to-end SSE test.
- A **real-socket TTFT test**: first content arrives in under ⅕ of the total stream time. httpx's in-process ASGI transport holds the whole response, so this test runs on real uvicorn servers.

## Task 4: Rate limiter and failover router (SQLite)

```bash
.venv/bin/python task4_router/mock_providers.py    # :9201  /{name}/{ok|429|slow|500|400|garbage}
PRIMARY_URL=http://127.0.0.1:9201/primary/429 .venv/bin/python task4_router/app.py   # :9200
curl -s localhost:9200/v1/chat/completions -H 'Authorization: Bearer sk-tenant-a' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

**Rate limiter** (`ratelimit.py`) is an exact sliding **log** in on-disk SQLite (WAL mode), with one row per admitted request.
- **Reserve:** inside a single `BEGIN IMMEDIATE` transaction it evicts expired rows, sums usage, and inserts the new row. Check-and-insert is therefore atomic across coroutines, threads **and processes**. The tests run 40 concurrent requests (exactly 10 admitted), and two DB connections × 4 threads (exactly 25 admitted).
- **Reserve, then reconcile or release:** the reservation is pessimistic (prompt estimate + `max_tokens`). It is replaced by the real `usage.total_tokens` when the reply arrives, and fully refunded if every provider fails.
- **Eviction:** expired rows are deleted per tenant on every reserve; `purge_expired()` sweeps idle tenants. `Retry-After` is exact: it walks rows oldest-first until enough tokens would expire.
- **Key storage:** tenant API keys are stored only as SHA-256 digests.

**Router** (`router.py`)
- Each attempt is wrapped in `asyncio.timeout(3.0)`, covering connect, send *and* body read. When the deadline fires, the in-flight request is cancelled, so a late primary reply can never race with or be double-counted against the secondary.
- **Fails over** on 429 and timeout (required by the brief), and also on 5xx, network errors and unparseable bodies.
- **Does not fail over** on other 4xx, since that's the caller's error, and the backup would reject it too.
- A small **circuit breaker** skips a primary that keeps failing, so callers don't pay the 3 s timeout on every request.

**Error sanitization** (`app.py`): every error the gateway emits has exactly this shape:
`{"error": {"type", "message", "request_id"}}`
- Error types: `authentication_error` 401, `invalid_request` 400, `request_too_large` 413, `rate_limit_exceeded` 429 (+ `Retry-After`), `upstream_rejected` 400, `upstream_unavailable` 503, `internal_error` 500.
- The tests assert that no `traceback`, internal hostnames, provider names or URLs appear in error bodies. Upstream details are logged server-side, tied to `request_id`.
- Successful responses are also rebuilt from allowlisted fields only (provider-internal fields are dropped).

## What I'd add for production
- **Task 2:** JWKS/RS256 or OAuth token introspection, per-tool policy (OPA/Cedar) instead of a prefix rule, optional filtering of `admin_*` tools out of `tools/list` for viewers, and audit logs.
- **Task 3:** redact the *request* path too, plus NER-based detection for names and addresses (run asynchronously, with the regex as the synchronous fast path).
- **Task 4:** Redis/Lua or Postgres for multi-node deployments (SQLite is only right for a single host), streaming support with a TTFT-based deadline, and hedged requests.
