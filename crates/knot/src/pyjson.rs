//! Python-compatible JSON text rendering (Laya `common.py` / `agent.py`).
//!
//! Laya renders `state`, criterion values and non-string `instructions` with
//! `json.dumps(..., ensure_ascii=False)` and the default `(", ", ": ")`
//! separators — spaces after commas and colons, raw UTF-8, and Python's
//! shortest float repr. serde_json's compact output (`{"a":1}`) and its float
//! formatting (`1e-5` where Python writes `1e-05`) diverge from that, so the
//! prompt half of a request must be rendered through here to stay byte-identical
//! to what the model was shown in Laya.

use serde_json::Value;

/// `json.dumps(value, ensure_ascii=False)` with Python's default separators.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

/// Python `str(value)` for scalar labels: strings pass through, `True`/`False`
/// (not `true`), integers without a decimal point, floats through `repr`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                float_repr(n.as_f64().unwrap_or(0.0))
            }
        }
        other => dumps(other),
    }
}

/// Python `repr` of a finite float: shortest round-trip digits, fixed notation
/// for decimal exponents in [-4, 16), otherwise `d[.ddd]e±NN` (sign always,
/// at least two exponent digits — Python writes `1e-05`, serde writes `1e-5`).
pub fn float_repr(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let sci = format!("{:e}", v); // shortest digits, one digit before the dot
    let (sign, rest) = match sci.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", sci.as_str()),
    };
    let (mantissa, exp) = rest.split_once('e').expect("{:e} always emits an exponent");
    let exp: i32 = exp.parse().expect("valid exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let mut out = String::from(sign);
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() >= int_len {
                out.push_str(&digits[..int_len]);
                let frac = &digits[int_len..];
                if frac.is_empty() {
                    out.push_str(".0");
                } else {
                    out.push('.');
                    out.push_str(frac);
                }
            } else {
                out.push_str(&digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                out.push_str(".0");
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
        out
    } else {
        let mut out = String::from(sign);
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.unsigned_abs()));
        out
    }
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(0.0)));
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(k, out);
                out.push_str(": ");
                write_value(v, out);
            }
            out.push('}');
        }
    }
}

/// Python's json encoder escapes only `"`, `\`, and controls < 0x20 (with
/// `\\b\\t\\n\\f\\r` for the named ones, `\\u00xx` lowercase otherwise);
/// everything else — including non-ASCII — passes through raw under
/// `ensure_ascii=False`.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Golden outputs captured from `python3 -c 'import json; ...'`.
    #[test]
    fn dumps_matches_python_default_separators() {
        assert_eq!(
            dumps(&json!({"a": [1, 2], "b": {"c": 1.0}})),
            r#"{"a": [1, 2], "b": {"c": 1.0}}"#
        );
        assert_eq!(dumps(&json!({"k": "héllo"})), r#"{"k": "héllo"}"#);
        assert_eq!(dumps(&json!([true, false, null])), "[true, false, null]");
        assert_eq!(dumps(&json!("x\ny\"z")), r#""x\ny\"z""#);
        assert_eq!(dumps(&json!("\u{1}")), r#""\u0001""#);
        assert_eq!(dumps(&json!("\u{7f}")), "\"\u{7f}\""); // DEL is not escaped
    }

    #[test]
    fn float_repr_matches_python() {
        for (v, want) in [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (3.0, "3.0"),
            (1.5, "1.5"),
            (0.0001, "0.0001"),
            (0.0001234, "0.0001234"),
            (1e-5, "1e-05"),
            (-1.5e-7, "-1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (-2.5e20, "-2.5e+20"),
            (1234.5, "1234.5"),
        ] {
            assert_eq!(float_repr(v), want, "float {v:e}");
        }
        assert_eq!(float_repr(-0.0), "-0.0");
        assert_eq!(float_repr(0.0), "0.0");
    }

    #[test]
    fn py_str_matches_python_str() {
        assert_eq!(py_str(&json!("red")), "red");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(false)), "False");
        assert_eq!(py_str(&json!(1)), "1");
        assert_eq!(py_str(&json!(1.0)), "1.0");
        assert_eq!(py_str(&json!(1.5)), "1.5");
    }
}
