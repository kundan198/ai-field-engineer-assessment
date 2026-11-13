import json

import httpx
import pytest

from task2_mcp_gateway import mock_downstream
from task2_mcp_gateway.gateway import create_app
from task2_mcp_gateway.tokens import mint

SECRET = "test-secret"
ADMIN = mint({"sub": "alice", "role": "admin"}, SECRET)
VIEWER = mint({"sub": "bob", "role": "viewer"}, SECRET)


@pytest.fixture
async def gw():
    mock_downstream.CALLS.clear()
    downstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_downstream.app))
    app = create_app("http://downstream/mcp", SECRET, client=downstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c
    await downstream.aclose()


def call(name, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": {}}}


async def post(c, body, token=VIEWER, raw=False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if raw:
        return await c.post("/mcp", content=body, headers={**headers, "content-type": "application/json"})
    return await c.post("/mcp", json=body, headers=headers)


async def test_tools_list_forwarded_transparently(gw):
    r = await post(gw, {"jsonrpc": "2.0", "id": 7, "method": "tools/list"})
    assert r.status_code == 200
    assert r.json()["id"] == 7
    assert {t["name"] for t in r.json()["result"]["tools"]} == {"get_customer_record", "admin_reset_key"}
    assert len(mock_downstream.CALLS) == 1


async def test_viewer_can_call_non_admin_tool(gw):
    r = await post(gw, call("get_customer_record"))
    assert r.json()["result"]["content"][0]["text"] == "ran get_customer_record"


@pytest.mark.parametrize("name", ["admin_reset_key", "ADMIN_reset_key", "  admin_reset_key", "Admin_x"])
async def test_viewer_blocked_from_admin_tools_without_downstream_call(gw, name):
    r = await post(gw, call(name, id_="abc"))
    body = r.json()
    assert body == {"jsonrpc": "2.0", "id": "abc", "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
    assert mock_downstream.CALLS == []  # never reached the server


async def test_admin_can_call_admin_tool(gw):
    r = await post(gw, call("admin_reset_key"), token=ADMIN)
    assert r.json()["result"]["content"][0]["text"] == "ran admin_reset_key"


async def test_token_not_passed_downstream_identity_is(gw):
    await post(gw, call("get_customer_record"))
    h = mock_downstream.CALLS[0]["headers"]
    assert "authorization" not in h
    assert h["x-gateway-role"] == "viewer" and h["x-gateway-user"] == "bob"


@pytest.mark.parametrize("token", [None, "garbage", mint({"sub": "x", "role": "admin"}, "wrong-secret"),
                                   mint({"sub": "x", "role": "admin"}, SECRET, ttl_s=-120)])
async def test_bad_tokens_rejected(gw, token):
    r = await post(gw, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token=token)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32001
    assert mock_downstream.CALLS == []


async def test_alg_none_rejected(gw):
    import base64
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    forged = f"{b({'alg': 'none'})}.{b({'role': 'admin', 'exp': 9999999999})}."
    r = await post(gw, call("admin_reset_key"), token=forged)
    assert r.status_code == 401


async def test_unknown_role_rejected(gw):
    r = await post(gw, call("x"), token=mint({"sub": "x", "role": "root"}, SECRET))
    assert r.status_code == 403


async def test_parse_error(gw):
    r = await post(gw, b"{not json", raw=True)
    assert r.json()["error"]["code"] == -32700


@pytest.mark.parametrize("body", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "1.0", "id": 1, "method": "x"},
                                  {"jsonrpc": "2.0", "id": 1, "method": 5}, "str", {"jsonrpc": "2.0", "id": {}, "method": "x"}])
async def test_invalid_request(gw, body):
    r = await post(gw, body)
    assert r.json()["error"]["code"] == -32600


async def test_batch_smuggling_blocked(gw):
    r = await post(gw, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, call("admin_reset_key", 2)])
    assert r.json()["error"]["code"] == -32600
    assert mock_downstream.CALLS == []


async def test_duplicate_key_smuggling_blocked(gw):
    raw = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_customer_record"}}'
    r = await post(gw, raw, raw=True)
    assert r.json()["error"]["code"] == -32600
    assert mock_downstream.CALLS == []


@pytest.mark.parametrize("params", [None, {}, {"name": 123}, {"name": ["admin_x"]}, {"name": "  "}])
async def test_malformed_tool_name(gw, params):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    if params is not None:
        body["params"] = params
    r = await post(gw, body)
    assert r.json()["error"]["code"] == -32602
    assert mock_downstream.CALLS == []


async def test_downstream_down_is_sanitised():
    app = create_app("http://127.0.0.1:1/mcp", SECRET)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await post(c, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert r.status_code == 502
    assert r.json() == {"jsonrpc": "2.0", "id": 3, "error": {"code": -32603, "message": "Downstream MCP server unavailable"}}
