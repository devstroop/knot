//! ORT logits golden for the knot-zig canary (phase 1).
//!
//! Mirrors knot's ONNX path exactly (crates/knot/src/runtime.rs):
//! - Session::builder() defaults + commit_from_file(laya.onnx) — no option
//!   overrides, no thread overrides (knot only sets intra threads when
//!   KNOT_ORT_INTRA_THREADS is set; it is unset here).
//! - Inputs by knot's names/dtypes/shapes: input_ids i64[1,L],
//!   attention_mask i64[1,L], marker_pos i64[1,M], marker_mask bool[1,M],
//!   qtype i64[1] -> output "logits" f32.
//!
//! The zig side feeds the SAME hardcoded values; both write raw f32 bytes,
//! compared via sha256sum in the shell.

use ort::value::Tensor;

const MODEL: &str = "/root/.cache/knot/english/laya.onnx";
// Fixed inputs (shared verbatim with the zig side).
const IDS: [i64; 16] = [
    10795, 253, 3634, 3662, 2634, 18037, 65, 32, // "Does the context answer `rust`?"
    510, 5570, 9300, 253, 2953, 285, 5783, 253, // "The agent updated the addre..."
];
const SEQ: usize = 16;
const NUM_MARKERS: usize = 2;
const MARKER_POS: [i64; 2] = [4, 12];
const MARKER_MASK: [bool; 2] = [true, true];
const QTYPE: [i64; 1] = [0];

fn main() {
    let mut session = ort::session::Session::builder()
        .expect("builder")
        .commit_from_file(MODEL)
        .expect("commit_from_file");

    // Report the model's declared I/O so the zig side can assert equality.
    // (rc10's Input/Output expose name + type only; shapes come back from the
    // tensor we extract.)
    for inp in &session.inputs {
        println!("input  {:?} type={:?}", inp.name, inp.input_type);
    }
    for out in &session.outputs {
        println!("output {:?} type={:?}", out.name, out.output_type);
    }

    let mk = |data: Vec<i64>, shape: Vec<usize>| -> Tensor<i64> {
        Tensor::from_array((shape, data.into_boxed_slice())).expect("tensor")
    };
    let ids = mk(IDS.to_vec(), vec![1, SEQ]);
    let att = mk(vec![1i64; SEQ], vec![1, SEQ]);
    let mpos = mk(MARKER_POS.to_vec(), vec![1, NUM_MARKERS]);
    let mmask = Tensor::from_array((vec![1, NUM_MARKERS], MARKER_MASK.to_vec().into_boxed_slice()))
        .expect("mask tensor");
    let qt = mk(QTYPE.to_vec(), vec![1]);

    let mut out = session
        .run(ort::inputs![
            "input_ids" => &ids,
            "attention_mask" => &att,
            "marker_pos" => &mpos,
            "marker_mask" => &mmask,
            "qtype" => &qt,
        ])
        .expect("run");

    let (shape, data) = out
        .get_mut("logits")
        .expect("logits output")
        .try_extract_tensor::<f32>()
        .expect("extract f32");
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    std::fs::write("logits_rust.bin", bytes).expect("write logits_rust.bin");
    println!(
        "logits: shape={:?} elems={} f32[0..4]={:?}",
        dims,
        data.len(),
        &data[..4.min(data.len())]
    );
    println!("wrote logits_rust.bin ({} bytes)", bytes.len());
}
