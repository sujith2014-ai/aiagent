//! Build and sign `.cap` packages natively (needed for on-device learning: the device has no Python). Mirrors packages/capbuild.py;
//! a test and the Python suite check that the Rust runtime accepts what this produces and that untrusted signers are rejected.
use crate::capability::{InputStats, Manifest, RoutingHints};
use crate::package::CapPackage;
use crate::train_lite::softmax;
use anyhow::{anyhow, bail, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;

/// A device-local signing identity. The seed is supplied by the platform (app-private storage / keystore); it is never part of a package.
pub struct DeviceSigner { pub key_id: String, key: SigningKey }

impl DeviceSigner {
    pub fn from_seed_hex(key_id: &str, seed_hex: &str) -> Result<Self> {
        let b = hex::decode(seed_hex.trim())?; let seed: [u8; 32] = b.try_into().map_err(|_| anyhow!("seed must be 32 bytes (64 hex chars)"))?;
        Ok(Self { key_id: key_id.into(), key: SigningKey::from_bytes(&seed) })
    }
    pub fn public_b64(&self) -> String { B64.encode(self.key.verifying_key().to_bytes()) }
    pub fn public_bytes(&self) -> [u8; 32] { self.key.verifying_key().to_bytes() }
}

pub fn input_stats(xs: &[Vec<f32>]) -> InputStats {
    let d = xs.first().map(|x| x.len()).unwrap_or(0); let n = xs.len().max(1) as f32;
    let mean: Vec<f32> = (0..d).map(|j| xs.iter().map(|x| x[j]).sum::<f32>() / n).collect();
    InputStats { min: (0..d).map(|j| xs.iter().map(|x| x[j]).fold(f32::INFINITY, f32::min)).collect(), max: (0..d).map(|j| xs.iter().map(|x| x[j]).fold(f32::NEG_INFINITY, f32::max)).collect(),
        std: (0..d).map(|j| (xs.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f32>() / n).sqrt()).collect(), mean }
}

/// Temperature scaling on validation data: grid search (geometric, 1..4) minimising NLL.
/// Never sharpens (T >= 1): on a small, perfectly separable validation set NLL keeps falling as T -> 0, which would just manufacture overconfidence.
pub fn fit_temperature(logits: &[Vec<f32>], y: &[usize]) -> f32 {
    if logits.is_empty() { return 1.0; }
    let (mut best, mut bt) = (f64::INFINITY, 1.0f32);
    for i in 0..60 { let t = 4.0f32.powf(i as f32 / 59.0);
        let nll: f64 = logits.iter().zip(y).map(|(l, &c)| { let p = softmax(&l.iter().map(|v| v / t).collect::<Vec<_>>()); -(p[c].max(1e-12) as f64).ln() }).sum::<f64>() / logits.len() as f64;
        if nll < best { best = nll; bt = t; } }
    bt
}

pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = days + 719468; let era = z.div_euclid(146097); let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; let y = yoe + era * 400; let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153; let d = doy - (153 * mp + 2) / 5 + 1; let m = if mp < 10 { mp + 3 } else { mp - 9 }; let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn sha(b: &[u8]) -> String { hex::encode(Sha256::digest(b)) }

pub struct Spec<'a> {
    pub capability_id: &'a str, pub version: &'a str, pub description: &'a str, pub keywords: &'a [String], pub labels: &'a [String], pub device_caps: &'a [String],
    pub dependencies: &'a [String], pub provenance: Value, pub min_accuracy: f64, pub role: Option<&'a str>,
}

/// Assemble and sign a package from raw parts. `model` is the ONNX bytes; `tests` are (input, expected label) pairs shipped inside the package.
pub fn assemble(spec: &Spec, model: &[u8], params: usize, input_dim: usize, tests: &[(Vec<f32>, usize)], stats: &InputStats, temperature: f32, signer: &DeviceSigner) -> Result<Vec<u8>> {
    let tests_blob: Vec<u8> = tests.iter().map(|(x, y)| format!("{}\n", json!({"input": x, "expected": y}))).collect::<String>().into_bytes();
    let hints = serde_json::to_vec(&json!({"description": spec.description, "keywords": spec.keywords}))?;
    let files: Vec<(&str, Vec<u8>)> = vec![("model/model.onnx", model.to_vec()), ("routing/hints.json", hints), ("tests/cases.jsonl", tests_blob)];
    write_package(spec, files, params, input_dim, tests.len(), stats, temperature, signer, model)
}

fn write_package(spec: &Spec, files: Vec<(&str, Vec<u8>)>, params: usize, input_dim: usize, n_tests: usize, stats: &InputStats, temperature: f32, signer: &DeviceSigner, model: &[u8]) -> Result<Vec<u8>> {
    let contents: serde_json::Map<String, Value> = files.iter().map(|(n, b)| (n.to_string(), json!(sha(b)))).collect();
    let mut m = json!({
        "format_version": "cap/1", "package_id": format!("{}-{}-{}", spec.capability_id, spec.version, &sha(model)[..8]), "capability_id": spec.capability_id, "version": spec.version,
        "model": {"format": "onnx", "variant": "fp32", "file": "model/model.onnx"},
        "io": {"input": {"dtype": "f32", "dim": input_dim, "description": "feature vector"}, "output": {"kind": "classification", "labels": spec.labels}},
        "requires": {"runtime_min": "0.1.0", "device_caps": spec.device_caps}, "dependencies": spec.dependencies,
        "resources": {"params": params, "model_bytes": model.len(), "min_ram_mb": 64}, "provenance": spec.provenance,
        "tests": {"file": "tests/cases.jsonl", "count": n_tests, "min_accuracy": spec.min_accuracy}, "routing_file": "routing/hints.json", "created": utc_now(),
        "signer": {"key_id": signer.key_id}, "contents": contents,
        "input_stats": {"min": stats.min, "max": stats.max, "mean": stats.mean, "std": stats.std}, "calibration": {"method": "temperature", "temperature": temperature},
    });
    if let Some(r) = spec.role { m["role"] = json!(r); }
    let manifest = serde_json::to_vec_pretty(&m)?;
    let sig = json!({"alg": "ed25519", "key_id": signer.key_id, "sig_b64": B64.encode(signer.key.sign(&manifest).to_bytes())});
    let mut zw = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    zw.start_file("manifest.json", opts)?; zw.write_all(&manifest)?;
    zw.start_file("signature.json", opts)?; zw.write_all(sig.to_string().as_bytes())?;
    for (n, b) in &files { zw.start_file(*n, opts)?; zw.write_all(b)?; }
    Ok(zw.finish()?.into_inner())
}

/// Router update without touching the model: same weights, tests and stats; extended routing keywords; new version; re-signed.
pub fn repackage_keywords(existing: &CapPackage, new_version: &str, extra_keywords: &[String], signer: &DeviceSigner) -> Result<Vec<u8>> {
    let old = &existing.manifest;
    if semver::Version::parse(new_version)? <= semver::Version::parse(&old.version)? { bail!("new version must be newer than {}", old.version); }
    let hints: RoutingHints = existing.hints().map_err(|e| anyhow!("{e}"))?;
    let mut kws = hints.keywords.clone(); for k in extra_keywords { if !kws.contains(k) { kws.push(k.clone()); } }
    let hints_blob = serde_json::to_vec(&json!({"description": hints.description, "keywords": kws}))?;
    let files: Vec<(&str, Vec<u8>)> = vec![("model/model.onnx", existing.model_bytes().to_vec()), ("routing/hints.json", hints_blob), ("tests/cases.jsonl", existing.files.get(&old.tests.file).cloned().unwrap_or_default())];
    let stats = old.input_stats.clone().ok_or_else(|| anyhow!("package has no input statistics"))?;
    let temp = old.calibration.as_ref().map(|c| c.temperature).unwrap_or(1.0);
    let mut prov = old.provenance.clone(); if !prov.is_object() { prov = json!({}); }
    prov["router_update"] = json!({"base_version": old.version, "added_keywords": extra_keywords, "signed_by": signer.key_id});
    let spec = Spec { capability_id: &old.capability_id, version: new_version, description: &hints.description, keywords: &kws, labels: &old.io.output.labels, device_caps: &old.requires.device_caps,
                      dependencies: &old.dependencies, provenance: prov, min_accuracy: old.tests.min_accuracy, role: old.role.as_deref() };
    write_package(&spec, files, old.resources.params as usize, old.io.input.dim, old.tests.count, &stats, temp, signer, existing.model_bytes())
}

pub fn manifest_of(bytes: &[u8]) -> Result<Manifest> { let tmp = std::env::temp_dir().join(format!("aicore-pk-{}.cap", sha(bytes).get(..16).unwrap_or("x"))); std::fs::write(&tmp, bytes)?; let p = CapPackage::open(&tmp).map_err(|e| anyhow!("{e}")); let _ = std::fs::remove_file(&tmp); Ok(p?.manifest) }
