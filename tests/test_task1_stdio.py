"""Drives the Task 1 server as a real subprocess over raw stdio and asserts:
   * every stdout line is a valid JSON-RPC 2.0 message (no log pollution)
   * logs go to stderr
   * validation failures map to -32602 with field-level data."""
import json
import pathlib
import subprocess
import sys

import pytest

SERVER = pathlib.Path(__file__).resolve().parents[1] / "task1_mcp_server" / "server.py"


def _call(id_, name, args):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": args}}


@pytest.fixture(scope="module")
def session():
    msgs = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        _call(2, "get_customer_record", {"customer_id": "CUST-00001"}),
        _call(3, "get_customer_record", {"customer_id": "cust-1"}),
        _call(4, "trigger_refund", {"customer_id": "CUST-00001", "amount": -5, "reason": "short"}),
        _call(5, "trigger_refund", {"customer_id": "CUST-00001", "amount": "10", "reason": "duplicate charge on card"}),
        _call(6, "trigger_refund", {"customer_id": "CUST-00001", "amount": 25.5, "reason": "duplicate charge on card"}),
        _call(7, "trigger_refund", {"customer_id": "CUST-00001", "amount": 1.999, "reason": "duplicate charge on card"}),
        _call(8, "trigger_refund", {"customer_id": "CUST-00001", "amount": 5, "reason": "          x", "extra": 1}),
        _call(9, "get_customer_record", {"customer_id": "CUST-99999"}),
        _call(10, "nope", {}),
        _call(11, "trigger_refund", {"customer_id": "CUST-00002", "amount": 9999, "reason": "too big for balance"}),
        _call(12, "trigger_refund", {"customer_id": "CUST-00001", "amount": True, "reason": "bool is not a float"}),
    ]
    p = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    for m in msgs:
        p.stdin.write(json.dumps(m) + "\n")
    p.stdin.flush()
    want = {m["id"] for m in msgs if "id" in m}
    parsed = []
    while want - {m.get("id") for m in parsed}:
        line = p.stdout.readline()
        assert line, "server closed stdout early"
        parsed.append(json.loads(line))  # raises if ANY stdout line isn't JSON
    rest, stderr = p.communicate(timeout=10)  # closes stdin -> server exits cleanly
    parsed += [json.loads(l) for l in rest.splitlines() if l.strip()]
    return {m["id"]: m for m in parsed if "id" in m}, parsed, stderr


def test_stdout_is_pure_jsonrpc(session):
    _, parsed, stderr = session
    assert all(m.get("jsonrpc") == "2.0" for m in parsed)
    assert "starting customer-ops MCP server" in stderr  # logs went to stderr


def test_tools_list(session):
    by_id, *_ = session
    names = {t["name"] for t in by_id[1]["result"]["tools"]}
    assert names == {"get_customer_record", "trigger_refund"}


def test_happy_path(session):
    by_id, *_ = session
    assert by_id[2]["result"]["structuredContent"]["name"] == "Ada Lovelace"
    assert by_id[6]["result"]["structuredContent"]["status"] == "queued"


@pytest.mark.parametrize("rid", [3, 4, 5, 7, 8, 12])
def test_invalid_params_are_jsonrpc_32602(session, rid):
    by_id, *_ = session
    err = by_id[rid]["error"]
    assert err["code"] == -32602
    assert isinstance(err["data"], list) and err["data"]


def test_multiple_field_errors_reported(session):
    by_id, *_ = session
    fields = {e["field"] for e in by_id[4]["error"]["data"]}
    assert {"amount", "reason"} <= fields


def test_unknown_tool(session):
    by_id, *_ = session
    assert by_id[10]["error"]["code"] == -32602


def test_business_errors_are_tool_results(session):
    by_id, *_ = session
    assert by_id[9]["result"]["isError"] is True
    assert by_id[11]["result"]["isError"] is True
