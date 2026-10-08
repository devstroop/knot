//! pyjson golden: render every corpus case through knot's own
//! `knot::pyjson::dumps` (the prompt-side renderer) and emit the strings for
//! the zig port to match byte-for-byte. serde_json must carry
//! `preserve_order` (it does, in this manifest) so object key order is the
//! corpus file's order — Python's insertion order.

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: pyjson_golden <cases.json> <rust_out.json>");
        std::process::exit(2);
    }
    let cases: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let mut out: Vec<serde_json::Value> = Vec::new();
    for (i, case) in cases.as_array().unwrap().iter().enumerate() {
        let rendered = knot::pyjson::dumps(case);
        out.push(serde_json::Value::String(rendered.clone()));
        // py_str coverage on scalar labels (bools/numbers/strings)
        if case.is_string() || case.is_boolean() || case.is_number() {
            let s = knot::pyjson::py_str(case);
            debug_assert_eq!(
                s,
                match case {
                    serde_json::Value::String(t) => t.clone(),
                    serde_json::Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
                    _ => rendered,
                },
                "py_str diverged at case {i}"
            );
        }
    }
    fs::write(&args[2], serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("rendered {} cases -> {}", out.len(), args[2]);
}
