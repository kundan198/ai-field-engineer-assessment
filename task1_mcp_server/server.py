"""Task 1: Customer-ops MCP server over stdio with strict Pydantic validation.

STDIO contract
  * stdout carries ONLY newline-delimited JSON-RPC 2.0 messages.
  * Every log / debug line goes to stderr via `logging` (never print()).
  * Defence in depth: while serving, the SDK's stdio_server() points fd 1 at
    stderr, so even a stray print() from a dependency cannot corrupt the wire.

Error contract
  * Malformed / out-of-range arguments  -> JSON-RPC error -32602 (Invalid params),
    with a machine-readable list of field errors in `error.data`.
  * Unknown tool name                    -> JSON-RPC error -32602.
  * Unexpected server fault              -> JSON-RPC error -32603 (Internal error),
    no stack trace on the wire (it is logged to stderr instead).
  * Business-rule failures (customer not found, refund exceeds balance) are
    *tool execution* errors -> CallToolResult(isError=true), per MCP spec, so the
    model can read and self-correct.
"""
from __future__ import annotations

import logging
import math
import sys
import uuid
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Callable

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

# ---------------------------------------------------------------- logging → stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("customer-ops-mcp")

# ---------------------------------------------------------------- schemas
CustomerId = Annotated[str, StringConstraints(pattern=r"^CUST-[0-9]{5}$", strict=True)]


class _Strict(BaseModel):
    # strict=True: no "123" -> 123 coercion, no True -> 1.0; extra="forbid": no unknown keys.
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class GetCustomerRecordArgs(_Strict):
    customer_id: CustomerId = Field(description="Customer identifier, format CUST-XXXXX (5 digits).")


MAX_REFUND = 10_000.00


class TriggerRefundArgs(_Strict):
    customer_id: CustomerId = Field(description="Customer identifier, format CUST-XXXXX (5 digits).")
    amount: float = Field(gt=0, le=MAX_REFUND, description="Refund amount, positive, max 2 decimal places.")
    reason: str = Field(min_length=10, max_length=500, description="Human-readable reason, >= 10 chars.")

    @field_validator("amount")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("amount must be finite")
        if Decimal(str(v)) != Decimal(str(v)).quantize(Decimal("0.01")):
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("reason must contain at least 10 non-whitespace-padded characters")
        return v.strip()


# ---------------------------------------------------------------- mock data store
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"customer_id": "CUST-00001", "name": "Ada Lovelace", "tier": "gold", "balance_paid": 1250.00},
    "CUST-00002": {"customer_id": "CUST-00002", "name": "Alan Turing", "tier": "silver", "balance_paid": 300.00},
}
REFUNDS: list[dict[str, Any]] = []


def _ok(payload: dict[str, Any]) -> types.CallToolResult:
    import json

    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(payload))], structured_content=payload
    )


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def get_customer_record(args: GetCustomerRecordArgs) -> types.CallToolResult:
    rec = CUSTOMERS.get(args.customer_id)
    if rec is None:
        return _tool_error(f"Customer {args.customer_id} not found")
    return _ok(rec)


def trigger_refund(args: TriggerRefundArgs) -> types.CallToolResult:
    rec = CUSTOMERS.get(args.customer_id)
    if rec is None:
        return _tool_error(f"Customer {args.customer_id} not found")
    already = sum(r["amount"] for r in REFUNDS if r["customer_id"] == args.customer_id)
    amount = float(Decimal(str(args.amount)).quantize(Decimal("0.01"), ROUND_HALF_UP))
    if already + amount > rec["balance_paid"]:
        return _tool_error(
            f"Refund of {amount:.2f} exceeds refundable balance "
            f"({rec['balance_paid'] - already:.2f}) for {args.customer_id}"
        )
    refund = {
        "refund_id": f"RF-{uuid.uuid4().hex[:12]}",
        "customer_id": args.customer_id,
        "amount": amount,
        "reason": args.reason,
        "status": "queued",
    }
    REFUNDS.append(refund)
    log.info("refund queued id=%s customer=%s amount=%.2f", refund["refund_id"], args.customer_id, amount)
    return _ok(refund)


# name -> (args model, implementation, description)
TOOLS: dict[str, tuple[type[_Strict], Callable[[Any], types.CallToolResult], str]] = {
    "get_customer_record": (GetCustomerRecordArgs, get_customer_record, "Fetch a customer record by ID."),
    "trigger_refund": (TriggerRefundArgs, trigger_refund, "Queue a refund for a customer."),
}


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.setdefault("additionalProperties", False)
    return schema


# ---------------------------------------------------------------- MCP handlers
async def handle_list_tools(ctx: Any, params: types.PaginatedRequestParams) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, input_schema=_json_schema(model))
            for name, (model, _, desc) in TOOLS.items()
        ]
    )


def _validation_error_data(err: ValidationError) -> list[dict[str, Any]]:
    # Return field path + message only; never echo the raw input back (could contain PII/injection).
    return [{"field": ".".join(str(p) for p in e["loc"]) or "<root>", "message": e["msg"], "type": e["type"]}
            for e in err.errors(include_url=False, include_input=False)]


async def handle_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    entry = TOOLS.get(params.name)
    if entry is None:
        raise MCPError(INVALID_PARAMS, f"Unknown tool: {params.name}")
    model, impl, _ = entry
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as e:
        log.warning("invalid params for %s: %d error(s)", params.name, e.error_count())
        raise MCPError(INVALID_PARAMS, f"Invalid params for tool '{params.name}'", data=_validation_error_data(e))
    try:
        return impl(args)
    except MCPError:
        raise
    except Exception:  # never leak a traceback onto the wire
        log.exception("tool %s crashed", params.name)
        raise MCPError(INTERNAL_ERROR, "Internal error")


server = Server("customer-ops", on_list_tools=handle_list_tools, on_call_tool=handle_call_tool)


async def main() -> None:
    log.info("starting customer-ops MCP server on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
