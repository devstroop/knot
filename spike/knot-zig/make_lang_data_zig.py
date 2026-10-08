#!/usr/bin/env python3
"""Generate zig/src/lang_data.zig from knot's lang_data.rs (single source of truth)."""
import re, sys

SRC = "/root/ai-workspace/knot/crates/knot/src/lang_data.rs"
DST = "/tmp/opencode/knot-zig-spike/zig/src/lang_data.zig"

text = open(SRC).read()

def zig_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

out = []
out.append("//! GENERATED from knot `crates/knot/src/lang_data.rs` by make_lang_data_zig.py — do not edit by hand.\n")

# &[&str] tables
for name, body in re.findall(r"pub static (\w+): &\[&str\] = &\[(.*?)\];", text, re.S):
    words = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    out.append(f"pub const {name}: []const []const u8 = &.{{\n")
    for w in words:
        out.append(f"    {zig_str(w)},\n")
    out.append("};\n\n")

# STOP_LANGS: &[(&str, &[&str])]
m = re.search(r"pub static STOP_LANGS: [^=]*= &\[(.*?)\];", text, re.S)
pairs = re.findall(r'\("(\w+)",\s*(\w+)\)', m.group(1))
out.append("pub const LangStop = struct { lang: []const u8, words: []const []const u8 };\n")
out.append("pub const STOP_LANGS: []const LangStop = &.{\n")
for lang, tbl in pairs:
    out.append(f"    .{{ .lang = {zig_str(lang)}, .words = {tbl} }},\n")
out.append("};\n\n")

# NON_EN_DIACRITICS: &str
m = re.search(r'pub static NON_EN_DIACRITICS: &str = "((?:[^"\\]|\\.)*)";', text)
out.append(f"pub const NON_EN_DIACRITICS: []const u8 = {zig_str(m.group(1))};\n\n")

# SCRIPT_RANGES: &[(&str, &[(u32, u32)])]
m = re.search(r"pub static SCRIPT_RANGES: [^=]*= &\[(.*?)\];", text, re.S)
scripts = re.findall(r'\("(\w+)",\s*&\[(.*?)\]\)', m.group(1), re.S)
out.append("pub const Range = struct { lo: u32, hi: u32 };\n")
out.append("pub const ScriptRange = struct { name: []const u8, ranges: []const Range };\n")
out.append("pub const SCRIPT_RANGES: []const ScriptRange = &.{\n")
for name, rbody in scripts:
    pairs2 = re.findall(r"\(0x([0-9a-fA-F]+),\s*0x([0-9a-fA-F]+)\)", rbody)
    out.append(f"    .{{ .name = {zig_str(name)}, .ranges = &.{{")
    out.append(", ".join(f".{{ .lo = 0x{a}, .hi = 0x{b} }}" for a, b in pairs2))
    out.append("} },\n")
out.append("};\n")

open(DST, "w").write("".join(out))

# sanity
ns = len(re.findall(r"pub static", text))
nz = open(DST).read().count("pub const")
print(f"tables in rust: {ns}; zig consts: {nz}; bytes: {len(''.join(out))}")
