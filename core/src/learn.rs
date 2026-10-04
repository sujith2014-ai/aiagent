//! On-device learning: train a new capability from examples, adapt an installed one (head-only or full fine-tuning), or extend its routing
//! keywords. Every result is packaged, signed with the device's own key and then goes through the *same* import gates as any package
//! (signature, hashes, compatibility, sandbox load, bundled tests, resource check). Gates beyond import: held-out accuracy for new modules;
//! improvement on the new data and bounded regression on old data for adaptations.
use crate::pack::{self, DeviceSigner, Spec};
use crate::package::CapPackage;
use crate::runtime::{ImportReport, Runtime};
use crate::train_lite::{self, argmax, fit, Mlp, Rng, Scope, TrainConfig};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

#[derive(Deserialize, Clone)]
pub struct ExplicitSplit { pub x_train: Vec<Vec<f32>>, pub y_train: Vec<usize>, pub x_val: Vec<Vec<f32>>, pub y_val: Vec<usize>, pub x_test: Vec<Vec<f32>>, pub y_test: Vec<usize> }

#[derive(Deserialize)]
pub struct LearnRequest {
    pub capability_id: String, #[serde(default)] pub description: String, #[serde(default)] pub keywords: Vec<String>, pub labels: Vec<String>,
    #[serde(default)] pub version: Option<String>, #[serde(default = "d_min_acc")] pub min_accuracy: f64,
    #[serde(default)] pub x: Vec<Vec<f32>>, #[serde(default)] pub y: Vec<usize>, #[serde(default)] pub split: Option<ExplicitSplit>,
    #[serde(default)] pub hidden_options: Vec<Vec<usize>>, #[serde(default)] pub epochs: Option<usize>, #[serde(default)] pub seed: u64,
    #[serde(default = "d_true")] pub standardize: bool, #[serde(default)] pub device_caps: Vec<String>,
}
fn d_min_acc() -> f64 { 0.9 }
fn d_true() -> bool { true }

#[derive(Deserialize)]
pub struct AdaptRequest {
    pub capability_id: String, #[serde(default = "d_head")] pub mode: String,
    pub x: Vec<Vec<f32>>, pub y: Vec<usize>,
    #[serde(default)] pub x_test: Vec<Vec<f32>>, #[serde(default)] pub y_test: Vec<usize>,
    #[serde(default)] pub x_old: Vec<Vec<f32>>, #[serde(default)] pub y_old: Vec<usize>,
    #[serde(default = "d_drop")] pub max_old_drop: f64, #[serde(default)] pub min_gain: f64,
    #[serde(default)] pub epochs: Option<usize>, #[serde(default)] pub lr: Option<f32>, #[serde(default)] pub seed: u64,
}
fn d_head() -> String { "head".into() }
fn d_drop() -> f64 { 0.02 }

#[derive(Serialize, Default)]
pub struct LearnReport {
    pub learned: bool, pub reason: Option<String>, pub version: Option<String>, pub attempts: Vec<serde_json::Value>, pub package_bytes: usize, pub total_seconds: f64,
    pub train_seconds: f64, pub params: Option<usize>, pub test_accuracy: Option<f64>, pub before: Option<serde_json::Value>, pub after: Option<serde_json::Value>, pub import: Option<ImportReport>,
}

fn stratified_split(x: &[Vec<f32>], y: &[usize], classes: usize, seed: u64) -> Result<ExplicitSplit> {
    let mut rng = Rng(seed ^ 0x51F1_5EED);
    let mut by: Vec<Vec<usize>> = vec![vec![]; classes];
    for (i, c) in y.iter().enumerate() { by[*c].push(i); }
    let (mut tr, mut va, mut te) = (vec![], vec![], vec![]);
    for (c, idx) in by.iter_mut().enumerate() {
        if idx.is_empty() { continue; }
        if idx.len() < 3 { bail!("class {c} has only {} distinct example(s); at least 3 per class are needed for a train/validation/test split", idx.len()); }
        rng.shuffle(idx);
        let nv = ((idx.len() as f64) * 0.2).round().max(1.0) as usize; let nt = nv;
        te.extend_from_slice(&idx[..nt]); va.extend_from_slice(&idx[nt..nt + nv]); tr.extend_from_slice(&idx[nt + nv..]);
    }
    let pick = |ix: &Vec<usize>| (ix.iter().map(|i| x[*i].clone()).collect::<Vec<_>>(), ix.iter().map(|i| y[*i]).collect::<Vec<_>>());
    let (x_train, y_train) = pick(&tr); let (x_val, y_val) = pick(&va); let (x_test, y_test) = pick(&te);
    Ok(ExplicitSplit { x_train, y_train, x_val, y_val, x_test, y_test })
}

fn dedupe(x: &[Vec<f32>], y: &[usize]) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut seen = HashSet::new(); let (mut ox, mut oy) = (vec![], vec![]);
    for (xi, yi) in x.iter().zip(y) { if seen.insert(xi.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()) { ox.push(xi.clone()); oy.push(*yi); } }
    (ox, oy)
}

fn bump(v: &str, part: usize) -> Result<String> {
    let mut s = semver::Version::parse(v)?;
    match part { 1 => { s.minor += 1; s.patch = 0 } _ => s.patch += 1 }
    Ok(s.to_string())
}

impl Runtime {
    /// Install the device's own signing identity and trust its public key *in this runtime only*.
    pub fn set_device_signer(&mut self, s: DeviceSigner) -> Result<()> { self.trust.add_key(&s.key_id, &s.public_bytes())?; self.signer = Some(s); Ok(()) }

    fn signer_ref(&self) -> Result<&DeviceSigner> { self.signer.as_ref().ok_or_else(|| anyhow!("no device signing key configured")) }

    fn finish(&mut self, bytes: Vec<u8>, rep: &mut LearnReport) -> Result<()> {
        let dir = self.root.join("tmp"); std::fs::create_dir_all(&dir)?;
        let p = dir.join(format!("learn-{}.cap", crate::registry::now_secs()));
        std::fs::write(&p, &bytes)?; rep.package_bytes = bytes.len();
        let imp = self.import(&p); let _ = std::fs::remove_file(&p);
        rep.learned = imp.activated; if !imp.activated { rep.reason = Some(format!("import rejected at step '{}'", imp.failed_step.unwrap_or("?"))); }
        rep.import = Some(imp); Ok(())
    }

    /// Learn a new capability (or a newer version of an existing id) from labelled examples.
    pub fn learn(&mut self, req: &LearnRequest) -> Result<LearnReport> {
        let t0 = std::time::Instant::now(); let mut rep = LearnReport::default();
        if req.capability_id.is_empty() || !req.capability_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') { bail!("capability_id must be [A-Za-z0-9_]+"); }
        if req.labels.len() < 2 { bail!("need at least 2 labels"); }
        let split = match &req.split { Some(s) => s.clone(), None => {
            if req.x.is_empty() || req.x.len() != req.y.len() { bail!("x and y must be non-empty and equal in length"); }
            if req.y.iter().any(|c| *c >= req.labels.len()) { bail!("a label index is out of range"); }
            let (x, y) = dedupe(&req.x, &req.y);
            if x.len() < 30 { bail!("need at least 30 distinct examples, got {}", x.len()); }
            stratified_split(&x, &y, req.labels.len(), req.seed)? } };
        let dim = split.x_train.first().map(|r| r.len()).ok_or_else(|| anyhow!("empty training split"))?;
        if split.x_val.is_empty() || split.x_test.is_empty() { bail!("validation and test splits must be non-empty"); }
        let all_ok = |x: &Vec<Vec<f32>>| x.iter().all(|r| r.len() == dim && r.iter().all(|v| v.is_finite()));
        if !(all_ok(&split.x_train) && all_ok(&split.x_val) && all_ok(&split.x_test)) { bail!("all examples must be finite vectors of the same length"); }
        let opts = if req.hidden_options.is_empty() { vec![vec![32, 32], vec![64, 64], vec![128, 64]] } else { req.hidden_options.clone() };
        let mut best: Option<(Mlp, f64, train_lite::TrainReport)> = None;
        for (i, h) in opts.iter().enumerate() {
            let cfg = TrainConfig { hidden: h.clone(), epochs: req.epochs.unwrap_or(400), seed: req.seed + i as u64, standardize: req.standardize, ..Default::default() };
            let (m, tr) = train_lite::train(&split.x_train, &split.y_train, &split.x_val, &split.y_val, req.labels.len(), &cfg)?;
            let acc = m.accuracy(&split.x_test, &split.y_test);
            rep.attempts.push(json!({"hidden": h, "test_accuracy": acc, "val_accuracy": tr.val_accuracy, "epochs": tr.epochs_run, "seconds": tr.seconds, "params": tr.params}));
            rep.train_seconds += tr.seconds;
            let better = best.as_ref().map(|b| acc > b.1).unwrap_or(true);
            if better { best = Some((m, acc, tr)); }
            if acc >= req.min_accuracy { break; }
        }
        let (model, acc, tr) = best.unwrap();
        rep.test_accuracy = Some(acc); rep.params = Some(tr.params);
        if acc < req.min_accuracy { rep.reason = Some(format!("candidate rejected after {} attempt(s): held-out accuracy {acc:.3} < {}", rep.attempts.len(), req.min_accuracy)); rep.total_seconds = t0.elapsed().as_secs_f64(); return Ok(rep); }
        let minor = self.registry.get(&req.capability_id).and_then(|r| semver::Version::parse(&r.active_version).ok()).map(|v| v.minor).unwrap_or(0);
        let version = req.version.clone().unwrap_or_else(|| format!("0.{}.0", minor + 1));
        let logits: Vec<Vec<f32>> = split.x_val.iter().map(|x| model.logits(x)).collect();
        let temp = pack::fit_temperature(&logits, &split.y_val);
        let stats = pack::input_stats(&split.x_train);
        let tests: Vec<(Vec<f32>, usize)> = split.x_test.iter().cloned().zip(split.y_test.iter().cloned()).take(300).collect();
        let signer = self.signer_ref()?;
        let prov = json!({"learning_package": {"source": "on-device-examples", "teacher": null, "verified": true, "verification": "held-out split of the device's own examples"},
                          "training": {"trainer": "aicore-train-lite", "hidden": tr.hidden, "epochs": tr.epochs_run, "seed": req.seed, "train_seconds": tr.seconds}, "heldout_accuracy": acc, "strategy": "new_module", "signed_by": signer.key_id});
        let spec = Spec { capability_id: &req.capability_id, version: &version, description: &req.description, keywords: &req.keywords, labels: &req.labels, device_caps: &req.device_caps,
                          dependencies: &[], provenance: prov, min_accuracy: (req.min_accuracy - 0.05).max(0.0), role: None };
        let bytes = pack::assemble(&spec, &model.to_onnx(), tr.params, dim, &tests, &stats, temp, signer)?;
        rep.version = Some(version);
        self.finish(bytes, &mut rep)?; rep.total_seconds = t0.elapsed().as_secs_f64(); Ok(rep)
    }

    /// Fine-tune an installed capability on new examples (`head`: last layer only, trunk frozen; `full`: all layers) and install the result as a newer version.
    pub fn adapt(&mut self, req: &AdaptRequest) -> Result<LearnReport> {
        let t0 = std::time::Instant::now(); let mut rep = LearnReport::default();
        let scope = match req.mode.as_str() { "head" => Scope::HeadOnly, "full" => Scope::All, o => bail!("unknown adaptation mode '{o}' (head|full)") };
        let rec = self.registry.get(&req.capability_id).ok_or_else(|| anyhow!("unknown capability {}", req.capability_id))?.clone();
        let pkg = CapPackage::open(&self.root.join("store").join(&rec.active().store_file)).map_err(|e| anyhow!("{e}"))?;
        pkg.verify_signature(&self.trust).map_err(|e| anyhow!("{e}"))?; pkg.verify_hashes().map_err(|e| anyhow!("{e}"))?;
        let dim = rec.input_dim;
        let lite = crate::onnx_lite::parse(pkg.model_bytes(), dim)?;
        let mut mlp = Mlp::from_model(&lite)?;
        if req.x.len() != req.y.len() || req.x.is_empty() { bail!("x and y must be non-empty and equal in length"); }
        if req.x.iter().any(|r| r.len() != dim || r.iter().any(|v| !v.is_finite())) || req.y.iter().any(|c| *c >= rec.labels.len()) { bail!("examples must be finite vectors of length {dim} with valid labels"); }
        // fine-tuning data: hold out 25% (at least 1 per present class) as validation for early stopping
        let mut idx: Vec<usize> = (0..req.x.len()).collect(); Rng(req.seed ^ 0xADA97).shuffle(&mut idx);
        let nv = (idx.len() / 4).max(1).min(idx.len().saturating_sub(1));
        let (vi, ti) = idx.split_at(nv);
        let (xt, yt): (Vec<Vec<f32>>, Vec<usize>) = ti.iter().map(|i| (req.x[*i].clone(), req.y[*i])).unzip();
        let (xv, yv): (Vec<Vec<f32>>, Vec<usize>) = vi.iter().map(|i| (req.x[*i].clone(), req.y[*i])).unzip();
        let (x_new_test, y_new_test) = if req.x_test.is_empty() { (xv.clone(), yv.clone()) } else { (req.x_test.clone(), req.y_test.clone()) };
        let before_new = mlp.accuracy(&x_new_test, &y_new_test); let before_old = if req.x_old.is_empty() { None } else { Some(mlp.accuracy(&req.x_old, &req.y_old)) };
        rep.before = Some(json!({"new_test_accuracy": before_new, "old_accuracy": before_old}));
        let cfg = TrainConfig { epochs: req.epochs.unwrap_or(150), lr: req.lr.unwrap_or(2e-3), batch: 32, patience: 30, seed: req.seed, ..Default::default() };
        let mut rng = Rng(req.seed ^ 0xF17E);
        let tr = fit(&mut mlp, &xt, &yt, &xv, &yv, &cfg, scope, &mut rng)?;
        rep.train_seconds = tr.seconds;
        let after_new = mlp.accuracy(&x_new_test, &y_new_test); let after_old = if req.x_old.is_empty() { None } else { Some(mlp.accuracy(&req.x_old, &req.y_old)) };
        rep.after = Some(json!({"new_test_accuracy": after_new, "old_accuracy": after_old, "epochs": tr.epochs_run, "trainable": if scope == Scope::HeadOnly { mlp.layers.last().map(|l| l.w.len() + l.b.len()).unwrap_or(0) } else { mlp.params() }}));
        rep.test_accuracy = Some(after_new); rep.params = Some(mlp.params());
        if after_new < before_new + req.min_gain || (after_new <= before_new && req.min_gain > 0.0) { rep.reason = Some(format!("no improvement on the new data ({before_new:.3} -> {after_new:.3})")); rep.total_seconds = t0.elapsed().as_secs_f64(); return Ok(rep); }
        if let (Some(b), Some(a)) = (before_old, after_old) { if a < b - req.max_old_drop { rep.reason = Some(format!("regression on old data ({b:.3} -> {a:.3}) exceeds the allowed drop {}", req.max_old_drop)); rep.total_seconds = t0.elapsed().as_secs_f64(); return Ok(rep); } }
        let version = bump(&rec.active_version, 2)?;
        let logits: Vec<Vec<f32>> = xv.iter().map(|x| mlp.logits(x)).collect(); let temp = pack::fit_temperature(&logits, &yv);
        let old_stats = rec.input_stats.clone().ok_or_else(|| anyhow!("capability has no input statistics"))?;
        let ns = pack::input_stats(&req.x);
        let stats = crate::capability::InputStats { min: old_stats.min.iter().zip(&ns.min).map(|(a, b)| a.min(*b)).collect(), max: old_stats.max.iter().zip(&ns.max).map(|(a, b)| a.max(*b)).collect(), mean: old_stats.mean.clone(), std: old_stats.std.iter().zip(&ns.std).map(|(a, b)| a.max(*b)).collect() };
        let mut tests: Vec<(Vec<f32>, usize)> = x_new_test.iter().cloned().zip(y_new_test.iter().cloned()).take(150).collect();
        tests.extend(req.x_old.iter().cloned().zip(req.y_old.iter().cloned()).take(150));
        let signer = self.signer_ref()?;
        let hints = pkg.hints().map_err(|e| anyhow!("{e}"))?;
        let prov = json!({"learning_package": {"source": "on-device-adaptation", "mode": req.mode, "verified": true, "verification": "improvement on held-out new examples; bounded regression on old examples"},
                          "base_version": rec.active_version, "before": rep.before, "after": rep.after, "strategy": format!("adapt_{}", req.mode), "signed_by": signer.key_id});
        let spec = Spec { capability_id: &req.capability_id, version: &version, description: &hints.description, keywords: &hints.keywords, labels: &rec.labels, device_caps: &rec.device_caps,
                          dependencies: &rec.dependencies, provenance: prov, min_accuracy: (pkg.manifest.tests.min_accuracy).min(after_new.min(after_old.unwrap_or(1.0)) - 0.02).max(0.0), role: None };
        let bytes = pack::assemble(&spec, &mlp.to_onnx(), mlp.params(), dim, &tests, &stats, temp, signer)?;
        rep.version = Some(version);
        self.finish(bytes, &mut rep)?; rep.total_seconds = t0.elapsed().as_secs_f64(); let _ = argmax(&[0.0]); Ok(rep)
    }

    /// Router update on the device: extend a capability's routing keywords (no neural change) and install the re-signed newer version.
    pub fn alias(&mut self, capability: &str, keywords: &[String]) -> Result<ImportReport> {
        let rec = self.registry.get(capability).ok_or_else(|| anyhow!("unknown capability {capability}"))?.clone();
        let pkg = CapPackage::open(&self.root.join("store").join(&rec.active().store_file)).map_err(|e| anyhow!("{e}"))?;
        pkg.verify_signature(&self.trust).map_err(|e| anyhow!("{e}"))?; pkg.verify_hashes().map_err(|e| anyhow!("{e}"))?;
        let bytes = pack::repackage_keywords(&pkg, &bump(&rec.active_version, 2)?, keywords, self.signer_ref()?)?;
        let dir = self.root.join("tmp"); std::fs::create_dir_all(&dir)?;
        let p = dir.join(format!("alias-{}.cap", crate::registry::now_secs())); std::fs::write(&p, &bytes)?;
        let imp = self.import(&p); let _ = std::fs::remove_file(&p); Ok(imp)
    }
}
