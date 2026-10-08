//! Golden generator for the tokenizer-parity spike.
//!
//! Uses the exact crate knot uses (`tokenizers` 0.21.4) with knot's exact
//! call semantics: `encode(text, add_special_tokens = false)` (see
//! knot/crates/knot/src/prompt.rs `encode_ids`). Emits goldens.json:
//! [{"s": <input>, "ids": [<token ids>]}].

use std::fs;
use std::process;

use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: goldens <tokenizer.json> <strings.json> <goldens.json>");
        process::exit(2);
    }
    let tokenizer = Tokenizer::from_file(&args[1]).expect("load tokenizer.json");
    let strings: Vec<String> = serde_json::from_str(&fs::read_to_string(&args[2]).unwrap())
        .expect("parse strings.json");

    let mut out = Vec::with_capacity(strings.len());
    let mut failures = 0usize;
    for s in &strings {
        match tokenizer.encode(s.as_str(), false) {
            Ok(enc) => out.push(serde_json::json!({
                "s": s,
                "ids": enc.get_ids(),
                "tokens": enc.get_tokens(),
                "offsets": enc.get_offsets(),
            })),
            Err(e) => {
                eprintln!("ENCODE FAILED for {s:?}: {e}");
                failures += 1;
            }
        }
    }
    assert_eq!(failures, 0, "{failures} encodes failed");
    fs::write(&args[3], serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!(
        "goldens: {} strings -> {}",
        out.len(),
        args[3]
    );
}
