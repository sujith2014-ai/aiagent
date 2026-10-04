//! mlp-lite backend: numerical parity with ONNX Runtime (golden logits computed by onnxruntime in Python) and robustness on hostile bytes.
use aicore::backend::{Backend, Model};
use aicore::onnx_lite::{parse, MlpLiteBackend};

const COMPARE: &[u8] = include_bytes!("fixtures/mlp_compare.onnx");
const NORMALIZED: &[u8] = include_bytes!("fixtures/mlp_normalized_tumor.onnx");
const GOLDEN: &str = include_str!("fixtures/golden.json");

fn golden(name: &str) -> (usize, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let v: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    let g = &v[name];
    let conv = |x: &serde_json::Value| -> Vec<Vec<f32>> { x.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|f| f.as_f64().unwrap() as f32).collect()).collect() };
    (g["dim"].as_u64().unwrap() as usize, conv(&g["inputs"]), conv(&g["logits"]))
}

fn parity(bytes: &[u8], name: &str) {
    let (dim, xs, want) = golden(name);
    let m = parse(bytes, dim).unwrap();
    let mut worst = 0f32;
    for (x, w) in xs.iter().zip(&want) {
        let got = m.run(x).unwrap();
        assert_eq!(got.len(), w.len());
        for (a, b) in got.iter().zip(w) { worst = worst.max((a - b).abs() / b.abs().max(1.0)); }
    }
    assert!(worst < 1e-4, "{name}: worst relative difference {worst}");
}

#[test] fn matches_onnxruntime_on_a_plain_mlp() { parity(COMPARE, "mlp_compare"); }
#[test] fn matches_onnxruntime_on_a_model_with_baked_in_standardisation() { parity(NORMALIZED, "mlp_normalized_tumor"); }

#[test]
fn rejects_wrong_input_dimension_and_wrong_run_length() {
    assert!(parse(COMPARE, 3).is_err());
    let m = parse(COMPARE, 2).unwrap();
    assert!(m.run(&[0.1]).is_err() && m.run(&[0.1, 0.2, 0.3]).is_err());
}

#[test]
fn backend_trait_rejects_other_formats() {
    assert!(MlpLiteBackend.load("tflite", COMPARE, 2).is_err());
    assert!(MlpLiteBackend.load("onnx", COMPARE, 2).is_ok());
}

/// xorshift: deterministic, no dependency
struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

#[test]
fn hostile_bytes_never_panic() {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for base in [COMPARE, NORMALIZED] {
        let dim = if base.len() == COMPARE.len() { 2 } else { 30 };
        let mut accepted = 0;
        for _ in 0..4000 {
            let mut b = base.to_vec();
            match rng.next() % 5 {
                0 => { let n = 1 + rng.next() % 4; for _ in 0..n { let i = (rng.next() as usize) % b.len(); b[i] ^= 1 << (rng.next() % 8); } }
                1 => { let n = (rng.next() as usize) % b.len(); b.truncate(n); }
                2 => { let i = (rng.next() as usize) % b.len(); b.insert(i, rng.next() as u8); }
                3 => { let i = (rng.next() as usize) % b.len(); b.remove(i); }
                _ => { let i = (rng.next() as usize) % b.len(); let j = (i + 1 + (rng.next() as usize) % 16).min(b.len()); for k in i..j { b[k] = rng.next() as u8; } }
            }
            let r = std::panic::catch_unwind(|| parse(&b, dim).and_then(|m| m.run(&vec![0.5; dim])));
            assert!(r.is_ok(), "parse/run panicked on mutated input");
            if let Ok(Ok(_)) = r { accepted += 1; }
        }
        assert!(accepted < 4000, "every mutation was accepted: the parser validates nothing");
    }
    for junk in [vec![], vec![0u8; 100], vec![0xffu8; 100], b"not onnx at all".to_vec()] {
        assert!(std::panic::catch_unwind(|| parse(&junk, 2)).is_ok());
    }
}
