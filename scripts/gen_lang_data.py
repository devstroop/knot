"""Generate crates/knot/src/lang_data.rs from laya/laya/lang.py data tables."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT.parent / "laya"))
from laya import lang

def rust_str_set(name, items, per_line=8):
    items = sorted(items)
    out = [f"pub static {name}: &[&str] = &["]
    for i in range(0, len(items), per_line):
        chunk = ", ".join('"%s"' % w.replace('"', '\\"') for w in items[i : i + per_line])
        out.append(f"    {chunk},")
    out.append("];")
    return "\n".join(out)

parts = ["//! Generated from laya/laya/lang.py by scripts/gen_lang_data.py — do not edit by hand.\n"]
parts.append(rust_str_set("STOP_EN", lang._STOP["en"]))
for lg in sorted(lang._STOP):
    if lg == "en":
        continue
    parts.append(rust_str_set(f"STOP_{lg.upper()}", lang._STOP[lg]))
parts.append(rust_str_set("SHORT_SWEDISH_WORDS", lang._SHORT_SWEDISH_WORDS))
parts.append('pub static STOP_LANGS: &[(&str, &[&str])] = &[')
for lg in sorted(lang._STOP):
    parts.append(f'    ("{lg}", STOP_{lg.upper()}),')
parts.append("];")
diac = "".join(sorted(lang._NON_EN_DIACRITICS))
parts.append(f'pub static NON_EN_DIACRITICS: &str = "{diac}";')
parts.append(rust_str_set("NORDIC_OVERLAP_WORDS", lang._NORDIC_OVERLAP_WORDS))
parts.append(rust_str_set("EN_COLLISION_WORDS", lang._EN_COLLISION_WORDS))

ranges = []
for name, rs in lang._SCRIPT_RANGES:
    ranges.append((name, [(lo, hi) for lo, hi in rs]))
parts.append("pub static SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[")
for name, rs in ranges:
    parts.append(f'    ("{name}", &[{", ".join(f"({lo:#x}, {hi:#x})" for lo, hi in rs)}]),')
parts.append("];")

with open(_REPO_ROOT / "crates/knot/src/lang_data.rs", "w") as f:
    f.write("\n\n".join(parts) + "\n")
print("wrote lang_data.rs")
