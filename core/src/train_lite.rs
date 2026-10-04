//! Pure-Rust trainer for the small MLPs this system uses (Gemm/Relu stacks with optional baked-in standardisation), plus an ONNX writer, so a
//! device with no ML framework can learn a capability locally. Dependency-free; deterministic for a given seed. Not a general framework:
//! dense layers, ReLU, softmax cross-entropy, Adam, mini-batches, early stopping on a validation split.
use crate::onnx_lite::MlpLiteModel;
use anyhow::{anyhow, bail, Result};

pub struct Layer { pub w: Vec<f32>, pub b: Vec<f32>, pub inp: usize, pub out: usize }
pub struct Mlp { pub in_dim: usize, pub layers: Vec<Layer>, pub norm: Option<(Vec<f32>, Vec<f32>)> }

#[derive(Clone, Debug)]
pub struct TrainConfig { pub hidden: Vec<usize>, pub epochs: usize, pub batch: usize, pub lr: f32, pub patience: usize, pub seed: u64, pub standardize: bool }
impl Default for TrainConfig { fn default() -> Self { Self { hidden: vec![32, 32], epochs: 400, batch: 128, lr: 5e-3, patience: 60, seed: 0, standardize: true } } }

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainReport { pub epochs_run: usize, pub val_accuracy: f64, pub train_accuracy: f64, pub params: usize, pub seconds: f64, pub hidden: Vec<usize> }

/// SplitMix64 + Box-Muller: tiny, deterministic, no dependency.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
    pub fn unit(&mut self) -> f32 { ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32 }
    pub fn normal(&mut self) -> f32 { let (u1, u2) = (self.unit(), self.unit()); (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() }
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
    pub fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { v.swap(i, self.below(i + 1)); } }
}

impl Mlp {
    pub fn new(in_dim: usize, hidden: &[usize], classes: usize, rng: &mut Rng) -> Self {
        let mut dims = vec![in_dim]; dims.extend_from_slice(hidden); dims.push(classes);
        let n = dims.len() - 1;
        let layers = (0..n).map(|i| { let (a, b) = (dims[i], dims[i + 1]); let sd = if i + 1 < n { (2.0 / a as f32).sqrt() } else { (1.0 / a as f32).sqrt() };
            Layer { w: (0..a * b).map(|_| rng.normal() * sd).collect(), b: vec![0.0; b], inp: a, out: b } }).collect();
        Self { in_dim, layers, norm: None }
    }
    pub fn params(&self) -> usize { self.layers.iter().map(|l| l.w.len() + l.b.len()).sum() }

    fn normalise(&self, x: &[f32]) -> Vec<f32> {
        match &self.norm { Some((m, s)) => x.iter().enumerate().map(|(i, v)| (v - m[i]) / s[i]).collect(), None => x.to_vec() }
    }
    /// activations after each layer input (acts[0] = normalised input); last element = logits
    fn forward_all(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let mut acts = vec![self.normalise(x)];
        for (i, l) in self.layers.iter().enumerate() {
            let a = acts.last().unwrap(); let mut o = vec![0f32; l.out];
            for j in 0..l.out { let mut s = l.b[j]; let row = &l.w[j * l.inp..(j + 1) * l.inp]; for k in 0..l.inp { s += row[k] * a[k]; } o[j] = if i + 1 < self.layers.len() && s < 0.0 { 0.0 } else { s }; }
            acts.push(o);
        }
        acts
    }
    pub fn logits(&self, x: &[f32]) -> Vec<f32> { self.forward_all(x).pop().unwrap() }
    pub fn predict(&self, x: &[f32]) -> usize { argmax(&self.logits(x)) }
    /// mean softmax cross-entropy
    pub fn loss(&self, xs: &[Vec<f32>], ys: &[usize]) -> f64 { if xs.is_empty() { return 0.0; } xs.iter().zip(ys).map(|(x, y)| -(softmax(&self.logits(x))[*y].max(1e-12) as f64).ln()).sum::<f64>() / xs.len() as f64 }
    pub fn accuracy(&self, xs: &[Vec<f32>], ys: &[usize]) -> f64 { if xs.is_empty() { return 0.0; } xs.iter().zip(ys).filter(|(x, y)| self.predict(x) == **y).count() as f64 / xs.len() as f64 }

    /// ONNX (opset 13, ir 7): [Sub, Div]? then Gemm(transB=1)/Relu stack; input "input" [1, in_dim], output "logits".
    pub fn to_onnx(&self) -> Vec<u8> {
        let mut nodes = vec![]; let mut inits = vec![]; let mut cur = "input".to_string();
        if let Some((m, s)) = &self.norm {
            inits.push(tensor("0.mean", &[m.len() as i64], m)); inits.push(tensor("0.std", &[s.len() as i64], s));
            nodes.push(node("Sub", &[&cur, "0.mean"], "n_sub", "n_sub", vec![])); nodes.push(node("Div", &["n_sub", "0.std"], "n_div", "n_div", vec![])); cur = "n_div".into();
        }
        for (i, l) in self.layers.iter().enumerate() {
            let (wn, bn) = (format!("w{i}"), format!("b{i}"));
            inits.push(tensor(&wn, &[l.out as i64, l.inp as i64], &l.w)); inits.push(tensor(&bn, &[l.out as i64], &l.b));
            let last = i + 1 == self.layers.len(); let out = if last { "logits".to_string() } else { format!("g{i}") };
            nodes.push(node("Gemm", &[&cur, &wn, &bn], &out, &format!("gemm{i}"), vec![attr_f("alpha", 1.0), attr_f("beta", 1.0), attr_i("transB", 1)])); cur = out;
            if !last { let r = format!("r{i}"); nodes.push(node("Relu", &[&cur, ], &r, &format!("relu{i}"), vec![])); cur = r; }
        }
        let mut graph = vec![]; for n in &nodes { put_msg(&mut graph, 1, n); } put_str(&mut graph, 2, "aicore-train-lite");
        for t in &inits { put_msg(&mut graph, 5, t); }
        put_msg(&mut graph, 11, &value_info("input", &[1, self.in_dim as i64])); put_msg(&mut graph, 12, &value_info("logits", &[1, self.layers.last().unwrap().out as i64]));
        let mut model = vec![]; put_varint_field(&mut model, 1, 7); put_str(&mut model, 2, "aicore-train-lite"); put_msg(&mut model, 7, &graph);
        let mut opset = vec![]; put_str(&mut opset, 1, ""); put_varint_field(&mut opset, 2, 13); put_msg(&mut model, 8, &opset);
        model
    }

    /// Rebuild an Mlp from an exported model (to fine-tune an installed capability).
    pub fn from_model(m: &MlpLiteModel) -> Result<Self> { m.to_mlp().ok_or_else(|| anyhow!("model is not a plain [Sub, Div]? + Gemm/Relu MLP; cannot adapt it on this device")) }
}

pub fn argmax(v: &[f32]) -> usize { v.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 }

pub fn softmax(v: &[f32]) -> Vec<f32> { let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect(); let s: f32 = e.iter().sum(); e.iter().map(|x| x / s).collect() }

/// What may be trained: every layer, or only the last (a "head" adapter: the trunk stays frozen).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scope { All, HeadOnly }

pub fn fit(mlp: &mut Mlp, xs: &[Vec<f32>], ys: &[usize], vx: &[Vec<f32>], vy: &[usize], cfg: &TrainConfig, scope: Scope, rng: &mut Rng) -> Result<TrainReport> {
    if xs.is_empty() || xs.len() != ys.len() { bail!("need matching, non-empty training examples"); }
    if xs.iter().any(|x| x.len() != mlp.in_dim || x.iter().any(|v| !v.is_finite())) { bail!("training inputs must be finite vectors of length {}", mlp.in_dim); }
    let t0 = std::time::Instant::now();
    let first = if scope == Scope::HeadOnly { mlp.layers.len() - 1 } else { 0 };
    let nl = mlp.layers.len();
    // Adam state per trainable layer
    let mut m: Vec<(Vec<f32>, Vec<f32>)> = mlp.layers.iter().map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()])).collect();
    let mut v: Vec<(Vec<f32>, Vec<f32>)> = mlp.layers.iter().map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()])).collect();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32); let mut step = 0i32;
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    let (mut best, mut best_loss, mut best_w, mut bad, mut epochs_run) = (-1.0f64, f64::INFINITY, snapshot(mlp), 0usize, 0usize);
    for _ in 0..cfg.epochs {
        epochs_run += 1; rng.shuffle(&mut idx);
        for chunk in idx.chunks(cfg.batch) {
            let mut gw: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect(); let mut gb: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
            for &i in chunk {
                let acts = mlp.forward_all(&xs[i]); let p = softmax(&acts[nl]);
                let mut delta: Vec<f32> = p.iter().enumerate().map(|(c, pc)| pc - if c == ys[i] { 1.0 } else { 0.0 }).collect();
                for li in (first..nl).rev() {
                    let l = &mlp.layers[li]; let a = &acts[li];
                    for j in 0..l.out { gb[li][j] += delta[j]; for k in 0..l.inp { gw[li][j * l.inp + k] += delta[j] * a[k]; } }
                    if li > first {
                        let mut nd = vec![0f32; l.inp];
                        for k in 0..l.inp { if acts[li][k] > 0.0 { let mut s = 0.0; for j in 0..l.out { s += l.w[j * l.inp + k] * delta[j]; } nd[k] = s; } }
                        delta = nd;
                    }
                }
            }
            step += 1; let scale = 1.0 / chunk.len() as f32; let (c1, c2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            for li in first..nl {
                for (i, g) in gw[li].iter().enumerate() { let g = g * scale; let mm = &mut m[li].0[i]; let vv = &mut v[li].0[i]; *mm = b1 * *mm + (1.0 - b1) * g; *vv = b2 * *vv + (1.0 - b2) * g * g; mlp.layers[li].w[i] -= cfg.lr * (*mm / c1) / ((*vv / c2).sqrt() + eps); }
                for (i, g) in gb[li].iter().enumerate() { let g = g * scale; let mm = &mut m[li].1[i]; let vv = &mut v[li].1[i]; *mm = b1 * *mm + (1.0 - b1) * g; *vv = b2 * *vv + (1.0 - b2) * g * g; mlp.layers[li].b[i] -= cfg.lr * (*mm / c1) / ((*vv / c2).sqrt() + eps); }
            }
        }
        let va = if vx.is_empty() { mlp.accuracy(xs, ys) } else { mlp.accuracy(vx, vy) };
        // accuracy is the criterion; ties are broken by validation loss, because small validation sets saturate accuracy at once and would freeze the weights at epoch 1
        let vl = if vx.is_empty() { mlp.loss(xs, ys) } else { mlp.loss(vx, vy) };
        if va > best + 1e-9 || (va >= best - 1e-9 && vl < best_loss - 1e-6) { best = best.max(va); best_loss = vl; best_w = snapshot(mlp); bad = 0; } else { bad += 1; if bad >= cfg.patience { break; } }
    }
    restore(mlp, &best_w);
    Ok(TrainReport { epochs_run, val_accuracy: best, train_accuracy: mlp.accuracy(xs, ys), params: mlp.params(), seconds: t0.elapsed().as_secs_f64(), hidden: cfg.hidden.clone() })
}

fn snapshot(m: &Mlp) -> Vec<(Vec<f32>, Vec<f32>)> { m.layers.iter().map(|l| (l.w.clone(), l.b.clone())).collect() }
fn restore(m: &mut Mlp, s: &[(Vec<f32>, Vec<f32>)]) { for (l, (w, b)) in m.layers.iter_mut().zip(s) { l.w.clone_from(w); l.b.clone_from(b); } }

/// Train a fresh MLP. Standardisation statistics come from the training split only and are baked into the model.
pub fn train(xs: &[Vec<f32>], ys: &[usize], vx: &[Vec<f32>], vy: &[usize], classes: usize, cfg: &TrainConfig) -> Result<(Mlp, TrainReport)> {
    let dim = xs.first().map(|x| x.len()).ok_or_else(|| anyhow!("no training examples"))?;
    let mut rng = Rng(cfg.seed ^ 0xA5A5_5A5A);
    let mut mlp = Mlp::new(dim, &cfg.hidden, classes, &mut rng);
    if cfg.standardize {
        let n = xs.len() as f32; let mean: Vec<f32> = (0..dim).map(|j| xs.iter().map(|x| x[j]).sum::<f32>() / n).collect();
        let std: Vec<f32> = (0..dim).map(|j| (xs.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f32>() / n).sqrt().max(1e-6)).collect();
        mlp.norm = Some((mean, std));
    }
    let rep = fit(&mut mlp, xs, ys, vx, vy, cfg, Scope::All, &mut rng)?;
    Ok((mlp, rep))
}

// ---- minimal protobuf encoder ---------------------------------------------------------------------------------------------
fn put_varint(b: &mut Vec<u8>, mut v: u64) { loop { let byte = (v & 0x7f) as u8; v >>= 7; if v == 0 { b.push(byte); break; } b.push(byte | 0x80); } }
fn put_tag(b: &mut Vec<u8>, field: u32, wt: u8) { put_varint(b, ((field as u64) << 3) | wt as u64); }
fn put_varint_field(b: &mut Vec<u8>, f: u32, v: u64) { put_tag(b, f, 0); put_varint(b, v); }
fn put_bytes(b: &mut Vec<u8>, f: u32, d: &[u8]) { put_tag(b, f, 2); put_varint(b, d.len() as u64); b.extend_from_slice(d); }
fn put_str(b: &mut Vec<u8>, f: u32, s: &str) { put_bytes(b, f, s.as_bytes()); }
fn put_msg(b: &mut Vec<u8>, f: u32, m: &[u8]) { put_bytes(b, f, m); }
fn tensor(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut t = vec![]; for d in dims { put_varint_field(&mut t, 1, *d as u64); } put_varint_field(&mut t, 2, 1); put_str(&mut t, 8, name);
    let mut raw = Vec::with_capacity(data.len() * 4); for x in data { raw.extend_from_slice(&x.to_le_bytes()); } put_bytes(&mut t, 9, &raw); t
}
fn attr_f(name: &str, v: f32) -> Vec<u8> { let mut a = vec![]; put_str(&mut a, 1, name); put_tag(&mut a, 2, 5); a.extend_from_slice(&v.to_le_bytes()); put_varint_field(&mut a, 20, 1); a }
fn attr_i(name: &str, v: i64) -> Vec<u8> { let mut a = vec![]; put_str(&mut a, 1, name); put_varint_field(&mut a, 3, v as u64); put_varint_field(&mut a, 20, 2); a }
fn node(op: &str, inputs: &[&str], output: &str, name: &str, attrs: Vec<Vec<u8>>) -> Vec<u8> {
    let mut n = vec![]; for i in inputs { put_str(&mut n, 1, i); } put_str(&mut n, 2, output); put_str(&mut n, 3, name); put_str(&mut n, 4, op); for a in &attrs { put_msg(&mut n, 5, a); } n
}
fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
    let mut shape = vec![]; for d in dims { let mut dim = vec![]; put_varint_field(&mut dim, 1, *d as u64); put_msg(&mut shape, 1, &dim); }
    let mut tt = vec![]; put_varint_field(&mut tt, 1, 1); put_msg(&mut tt, 2, &shape);
    let mut tp = vec![]; put_msg(&mut tp, 1, &tt);
    let mut vi = vec![]; put_str(&mut vi, 1, name); put_msg(&mut vi, 2, &tp); vi
}

#[cfg(test)]
mod tests {
    use super::*;
    fn blobs(n: usize, rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut xs = vec![]; let mut ys = vec![];
        for i in 0..n { let c = i % 3; let (cx, cy) = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.5)][c]; xs.push(vec![cx + rng.normal() * 0.6, cy + rng.normal() * 0.6, 100.0 + rng.normal()]); ys.push(c); }
        (xs, ys)
    }
    #[test] fn learns_separable_blobs_and_is_deterministic() {
        let mut r = Rng(1); let (x, y) = blobs(300, &mut r); let (vx, vy) = blobs(90, &mut r);
        let cfg = TrainConfig { epochs: 80, ..Default::default() };
        let (m, rep) = train(&x, &y, &vx, &vy, 3, &cfg).unwrap();
        assert!(rep.val_accuracy > 0.95, "val accuracy {}", rep.val_accuracy);
        let (m2, _) = train(&x, &y, &vx, &vy, 3, &cfg).unwrap();
        assert_eq!(m.to_onnx(), m2.to_onnx());                                   // same seed, same bytes
    }
    #[test] fn exported_onnx_roundtrips_through_the_mlp_lite_executor() {
        let mut r = Rng(2); let (x, y) = blobs(150, &mut r);
        let (m, _) = train(&x, &y, &[], &[], 3, &TrainConfig { epochs: 20, ..Default::default() }).unwrap();
        let lite = crate::onnx_lite::parse(&m.to_onnx(), 3).unwrap();
        use crate::backend::Model;
        for xi in x.iter().take(20) { let a = m.logits(xi); let b = lite.run(xi).unwrap(); for (p, q) in a.iter().zip(&b) { assert!((p - q).abs() < 1e-4 * q.abs().max(1.0), "{p} vs {q}"); } }
        let back = Mlp::from_model(&lite).unwrap();
        assert_eq!(back.layers.len(), m.layers.len()); assert!(back.norm.is_some());
    }
    #[test] fn head_only_training_leaves_the_trunk_untouched() {
        let mut r = Rng(3); let (x, y) = blobs(150, &mut r);
        let (mut m, _) = train(&x, &y, &[], &[], 3, &TrainConfig { epochs: 10, ..Default::default() }).unwrap();
        let trunk: Vec<Vec<f32>> = m.layers[..m.layers.len() - 1].iter().map(|l| l.w.clone()).collect();
        let last = m.layers.last().unwrap().w.clone();
        let mut rng = Rng(9); fit(&mut m, &x, &y, &[], &[], &TrainConfig { epochs: 5, ..Default::default() }, Scope::HeadOnly, &mut rng).unwrap();
        for (i, t) in trunk.iter().enumerate() { assert_eq!(&m.layers[i].w, t); }
        assert_ne!(m.layers.last().unwrap().w, last);
    }
    #[test] fn bad_inputs_are_refused() {
        assert!(train(&[], &[], &[], &[], 2, &TrainConfig::default()).is_err());
        assert!(train(&[vec![f32::NAN, 1.0]], &[0], &[], &[], 2, &TrainConfig::default()).is_err());
        assert!(train(&[vec![1.0, 1.0]], &[0, 1], &[], &[], 2, &TrainConfig::default()).is_err());
    }
}
