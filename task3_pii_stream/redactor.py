"""Streaming PII redactor with a bounded, *minimal* hold-back buffer.

Problem: an LLM streams "...mail me at jo" | "hn.doe@ex" | "ample.com today".
No single delta contains the email, so per-chunk regex leaks it; buffering the
whole response kills TTFT and memory.

Approach (per stream / per choice):
  * Append each delta to a small buffer.
  * Compute a safe cut point: everything before it can never become part of a
    future match, so it is redacted and emitted immediately. The rest is held.
  * The cut point is the later of
      a) start of the trailing "PII-shaped run". PII never spans whitespace,
         except single spaces/dashes *between digits* (cards like 4111 1111 …,
         SSNs like 123 45 6789). So normal prose like "Hello there " is
         released the moment the space arrives; only the last word is held.
      b) len(buf) - MAX_PII_LEN. Every pattern is length-bounded, so a match
         can never start further back than that. This caps the buffer (O(1)
         memory) even for adversarial input like a 10k-char unbroken token.
    …then pulled back to the start of any already-complete match that crosses it.
  * One char of already-emitted text is kept as left context so \b anchors
    behave exactly as they would on the full text.
Cost per delta is O(len(buf)) with buf <= MAX_PII_LEN + len(delta).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REDACTED = "[REDACTED]"

# All patterns are length-bounded so MAX_PII_LEN is a hard upper bound.
EMAIL = r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,4}\.[A-Za-z]{2,24}"
SSN = r"(?<!\d)(?!000|666|9\d\d)\d{3}(?P<sep>[- ])(?!00)\d{2}(?P=sep)(?!0000)\d{4}(?!\d)"
CARD = r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)"  # 13–19 digits, optional single separators; Luhn-checked
PATTERN = re.compile(f"(?P<email>{EMAIL})|(?P<ssn>{SSN})|(?P<card>{CARD})")
MAX_PII_LEN = 64 + 1 + 63 + 5 * 64 + 25  # longest possible EMAIL match (the longest pattern)

_RUN_CHAR = re.compile(r"[A-Za-z0-9._%+\-@]")


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _replace(m: re.Match) -> str:
    if m.lastgroup == "card":
        digits = re.sub(r"\D", "", m.group())
        return REDACTED if luhn_ok(digits) else m.group()
    return REDACTED


def _trailing_run_start(buf: str) -> int:
    """Index where the trailing run of characters that could still belong to a PII match begins."""
    i = len(buf)
    while i > 0:
        c = buf[i - 1]
        if _RUN_CHAR.match(c):
            i -= 1
        elif c in " -" and i >= 2 and buf[i - 2].isdigit() and (i == len(buf) or buf[i].isdigit()):
            i -= 1  # separator inside / at the tail of a digit group
        else:
            break
    return i


@dataclass
class StreamRedactor:
    max_hold: int = MAX_PII_LEN
    _buf: str = ""
    _ctx: str = ""  # last emitted char, used only as left context for anchors
    redactions: int = field(default=0)

    def _redact(self, text: str, end: int) -> str:
        """Redact matches fully inside text[:end] (text is ctx+buf)."""
        out, last = [], len(self._ctx)
        for m in PATTERN.finditer(text, len(self._ctx)):
            if m.start() >= end:
                break
            rep = _replace(m)
            if rep is REDACTED:
                self.redactions += 1
            out.append(text[last:m.start()])
            out.append(rep)
            last = m.end()
        out.append(text[last:end])
        return "".join(out)

    def feed(self, delta: str) -> str:
        """Add a delta; return whatever is now provably safe to emit (may be '')."""
        if not delta:
            return ""
        self._buf += delta
        text = self._ctx + self._buf
        off = len(self._ctx)
        cut = max(_trailing_run_start(self._buf), len(self._buf) - self.max_hold) + off
        # Never split a match that is already present and crosses the cut.
        for m in PATTERN.finditer(text, off):
            if m.start() >= cut:
                break
            if m.end() > cut:
                cut = m.start()
                break
        if cut <= off:
            return ""
        out = self._redact(text, cut)
        self._ctx = text[cut - 1]
        self._buf = text[cut:]
        return out

    def flush(self) -> str:
        """End of stream: redact and release everything still held."""
        text = self._ctx + self._buf
        out = self._redact(text, len(text))
        self._buf, self._ctx = "", ""
        return out

    @property
    def held(self) -> int:
        return len(self._buf)
