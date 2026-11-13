"""Token-aware sliding-window rate limiter persisted in on-disk SQLite.

Model: an exact sliding *log*: one row per admitted request (tenant, ts_ms, tokens).
  usage(tenant, now) = SUM(tokens) WHERE ts_ms > now - window

Request lifecycle (reserve → reconcile/release):
  1. reserve(tenant, estimated_tokens): atomically evict + sum + insert in ONE
     `BEGIN IMMEDIATE` transaction, so N concurrent requests (threads, coroutines,
     or separate gateway processes sharing the DB file) can never jointly exceed
     the limit. Estimate = prompt tokens + max_tokens (worst case).
  2. reconcile(id, actual_tokens): after the provider answers, replace the
     estimate with real `usage.total_tokens` (frees unused headroom).
  3. release(id): request failed upstream: refund fully.

Eviction: rows with ts_ms <= now - window are deleted for the tenant on every
reserve (keeps the per-tenant scan tiny), plus `purge_expired()` sweeps idle tenants.
Retry-After is exact: walk the tenant's rows oldest-first until enough tokens expire.
Tenant API keys are stored as SHA-256 digests, never in plaintext.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts_ms   INTEGER NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS ix_usage_tenant_ts ON token_usage (tenant, ts_ms);
CREATE INDEX IF NOT EXISTS ix_usage_ts ON token_usage (ts_ms);
"""


def tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reservation_id: int | None
    limit: int
    remaining: int
    retry_after_s: int = 0
    too_large: bool = False


class SlidingWindowTokenLimiter:
    def __init__(self, db_path: str, limit_tokens: int = 50_000, window_s: float = 60.0,
                 clock: Callable[[], float] = time.time):
        self.limit = limit_tokens
        self.window_ms = int(window_s * 1000)
        self.clock = clock
        self._lock = threading.Lock()  # serialise use of the shared connection inside this process
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=5.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    # ---------------------------------------------------------------- sync core
    def reserve_sync(self, tenant: str, tokens: int) -> Decision:
        tokens = max(0, int(tokens))
        with self._lock:
            cur = self._db.cursor()
            cur.execute("BEGIN IMMEDIATE")  # takes the write lock up front: check+insert is atomic across processes
            try:
                now = self._now_ms()
                cutoff = now - self.window_ms
                cur.execute("DELETE FROM token_usage WHERE tenant = ? AND ts_ms <= ?", (tenant, cutoff))
                used = cur.execute("SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?",
                                   (tenant,)).fetchone()[0]
                if tokens > self.limit:
                    cur.execute("COMMIT")
                    return Decision(False, None, self.limit, max(0, self.limit - used), 0, too_large=True)
                if used + tokens > self.limit:
                    need = used + tokens - self.limit
                    freed, retry_ms = 0, self.window_ms
                    for ts, t in cur.execute(
                            "SELECT ts_ms, tokens FROM token_usage WHERE tenant = ? ORDER BY ts_ms", (tenant,)):
                        freed += t
                        if freed >= need:
                            retry_ms = ts + self.window_ms - now
                            break
                    cur.execute("COMMIT")
                    return Decision(False, None, self.limit, max(0, self.limit - used),
                                    max(1, math.ceil(retry_ms / 1000)))
                cur.execute("INSERT INTO token_usage (tenant, ts_ms, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
                rid = cur.lastrowid
                cur.execute("COMMIT")
                return Decision(True, rid, self.limit, self.limit - used - tokens)
            except BaseException:
                cur.execute("ROLLBACK")
                raise

    def reconcile_sync(self, reservation_id: int, actual_tokens: int) -> None:
        with self._lock:
            self._db.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), reservation_id))

    def release_sync(self, reservation_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def usage_sync(self, tenant: str) -> int:
        with self._lock:
            return self._db.execute("SELECT COALESCE(SUM(tokens),0) FROM token_usage WHERE tenant=? AND ts_ms > ?",
                                    (tenant, self._now_ms() - self.window_ms)).fetchone()[0]

    def purge_expired(self) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM token_usage WHERE ts_ms <= ?",
                                    (self._now_ms() - self.window_ms,)).rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------------------------------------------------------------- async facade (keeps the event loop free)
    async def reserve(self, tenant: str, tokens: int) -> Decision:
        return await asyncio.to_thread(self.reserve_sync, tenant, tokens)

    async def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        await asyncio.to_thread(self.reconcile_sync, reservation_id, actual_tokens)

    async def release(self, reservation_id: int) -> None:
        await asyncio.to_thread(self.release_sync, reservation_id)
