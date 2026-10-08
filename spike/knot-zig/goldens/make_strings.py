#!/usr/bin/env python3
"""Golden corpus for the knot-zig tokenizer parity spike.

Shared by BOTH sides: the Rust generator (tokenizers 0.21.4, the crate knot
uses) reads strings.json and emits goldens.json; the zig port must match
every id list byte-for-byte. Edit via this script, never by hand.

Covers: ASCII prose, GPT2-regex branches (contractions, punctuation runs,
leading spaces), numbers, whitespace/newlines, non-Latin scripts (NFC),
and two explicit GAP probes (decomposed input; emoji) so the report can
quantify what the zig port does not yet handle instead of hiding it.
"""

import json
from pathlib import Path

STRINGS = [
    # --- ASCII prose / prompt-like ---
    "Does the context answer `rust`?",
    "The agent updated the address and confirmed the carrier has not received the parcel.",
    "Which id in `context` answers `query`? If none of them answers it, choose the closest one.",
    "",
    " ",
    "  leading and trailing   ",
    "tabs\there\tand\tthere",
    "new\nlines\nhere",
    "multiple     spaces    inside",
    # --- GPT2 pattern branches ---
    "it's a test — don't you think? I'd've said so; they're here (already).",
    "don'tSTOPCan'twon't",
    "3.14159 42 1,000,000 12/31/2026 $19.99 100%",
    "snake_case_identifier CamelCaseName kebab-case-name",
    "UPPER lower MiXeD",
    "!!!???...---___'''\"\"\"",
    "https://example.com/path?q=1&r=2#frag",
    # --- non-Latin, NFC-composed ---
    "Tack, nu fungerar det igen! Helt löst.",
    "الأرجو أن يصل الطلب غدا",
    "こんにちは、これはテストです。",
    "Привет, это тестовое сообщение.",
    "Δοκιμή πρωτοκόλλου συστήματος",
    "Die Straße nach München war frei.",
    # --- GAP probes (report must quantify, not hide) ---
    "cafe\u0301 decomposed: café",   # NFD: e + combining acute (explicit escape)
    "emoji probe: 🚀🔥 and 🇸🇪",       # byte-level emoji flags
    # --- boundaries ---
    "a",
    "0",
    " ",
    "\n",
    "tokenizers parity check 0123456789",
]


def main() -> None:
    out = Path(__file__).resolve().parent / "strings.json"
    out.write_text(json.dumps(STRINGS, ensure_ascii=False, indent=1))
    print(f"{len(STRINGS)} strings -> {out}")


if __name__ == "__main__":
    main()
