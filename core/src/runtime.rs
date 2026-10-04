//! Runtime: guarded import (the activation sequence), lazy load/unload, solve.
use crate::backend::{Backend, Model, OnnxBackend};
use crate::capability::Manifest;
use crate::device::DeviceProfile;
use crate::package::{CapPackage, PackageError};
use crate::registry::{now_secs, CapabilityRecord, Registry, Stats, VersionRecord};
use crate::router::{KeywordRouter, Router, Status, Task};
use crate::trace::Trace;
use crate::trust::TrustStore;
use crate::workspace::{Step, Workspace};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Serialize, Debug, Clone)]
pub struct StepResult { pub step: &'static str, pub ok: bool, pub detail: String }

#[derive(Serialize, Debug)]
pub struct ImportReport {
    pub activated: bool,
    pub package_id: Option<String>,
    pub capability_id: Option<String>,
    pub version: Option<String>,
    pub failed_step: Option<&'static str>,
    pub steps: Vec<StepResult>,
    pub test_accuracy: Option<f64>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "result")]
pub enum Outcome {
    #[serde(rename = "ANSWER")]
    Answer {
        status: Status, capability_id: String, version: String,
        label: String, label_index: usize, probs: Vec<f32>, confidence: f32,
        route_score: f64, latency_us: u64, task_id: String,
    },
    #[serde(rename = "NEEDS_HELP")]
    NeedsHelp { reason: String, task_id: String },
}

pub struct Runtime {
    root: PathBuf,
    pub device: DeviceProfile,
    trust: TrustStore,
    pub registry: Registry,
    backend: Box<dyn Backend>,
    router: Box<dyn Router + Send>,
    loaded: HashMap<String, (String, Box<dyn Model>)>, // capability -> (version, model)
    lru: Vec<String>,
    trace: Trace,
    task_counter: u64,
    pub conf_threshold: f32,
}

impl Runtime {
    pub fn open(root: &Path, device: DeviceProfile, trust: TrustStore) -> Result<Self> {
        std::fs::create_dir_all(root.join("store"))?;
        Ok(Self {
            root: root.to_path_buf(), device, trust,
            registry: Registry::open(root)?,
            backend: Box::new(OnnxBackend),
            router: Box::new(KeywordRouter::default()),
            loaded: HashMap::new(), lru: vec![],
            trace: Trace::open(root), task_counter: 0, conf_threshold: 0.6,
        })
    }

    pub fn loaded_modules(&self) -> Vec<String> { self.lru.clone() }

    pub fn unload(&mut self, cap: &str) -> bool {
        self.lru.retain(|c| c != cap);
        self.loaded.remove(cap).is_some()
    }

    fn fail(rep: &mut ImportReport, step: &'static str, e: impl ToString) {
        rep.steps.push(StepResult { step, ok: false, detail: e.to_string() });
        rep.failed_step = Some(step);
    }

    fn okstep(rep: &mut ImportReport, step: &'static str, detail: impl ToString) {
        rep.steps.push(StepResult { step, ok: true, detail: detail.to_string() });
    }

    /// parse -> signature -> hashes -> compat -> version policy -> sandbox load -> bundled tests -> resource check -> activate.
    /// Any failure leaves registry and store untouched.
    pub fn import(&mut self, path: &Path) -> ImportReport {
        let mut rep = ImportReport { activated: false, package_id: None, capability_id: None, version: None,
            failed_step: None, steps: vec![], test_accuracy: None };
        let pkg = match CapPackage::open(path) { Ok(p) => p, Err(e) => { Self::fail(&mut rep, "parse", e); return self.log_import(rep, path); } };
        rep.package_id = Some(pkg.manifest.package_id.clone());
        rep.capability_id = Some(pkg.manifest.capability_id.clone());
        rep.version = Some(pkg.manifest.version.clone());
        Self::okstep(&mut rep, "parse", "manifest readable");

        if let Err(e) = pkg.verify_signature(&self.trust) { Self::fail(&mut rep, "signature", e); return self.log_import(rep, path); }
        Self::okstep(&mut rep, "signature", format!("trusted signer {}", pkg.manifest.signer.key_id));

        if let Err(e) = pkg.verify_hashes() { Self::fail(&mut rep, "hashes", e); return self.log_import(rep, path); }
        Self::okstep(&mut rep, "hashes", format!("{} files verified", pkg.manifest.contents.len()));

        if let Err(e) = pkg.check_compat(&self.device, self.backend.formats()) { Self::fail(&mut rep, "compatibility", e); return self.log_import(rep, path); }
        Self::okstep(&mut rep, "compatibility", format!("device {}", self.device.name));

        if let Some(existing) = self.registry.get(&pkg.manifest.capability_id) {
            let have = semver::Version::parse(&existing.active().version).ok();
            let new = semver::Version::parse(&pkg.manifest.version).ok();
            if let (Some(h), Some(n)) = (have, new) {
                if n <= h { Self::fail(&mut rep, "version", format!("version {n} is not newer than active {h}")); return self.log_import(rep, path); }
            }
        }
        for dep in &pkg.manifest.dependencies {
            if self.registry.get(dep).is_none() { Self::fail(&mut rep, "dependencies", format!("missing dependency capability '{dep}'")); return self.log_import(rep, path); }
        }

        // sandbox load: model is loaded only through the backend, with a size sanity check
        let mb = pkg.model_bytes();
        if mb.len() as u64 != pkg.manifest.resources.model_bytes {
            Self::fail(&mut rep, "sandbox_load", "declared model_bytes differs from actual"); return self.log_import(rep, path);
        }
        let model = match self.backend.load(&pkg.manifest.model.format, mb, pkg.manifest.io.input.dim) {
            Ok(m) => m, Err(e) => { Self::fail(&mut rep, "sandbox_load", e); return self.log_import(rep, path); }
        };
        Self::okstep(&mut rep, "sandbox_load", self.backend.name());

        // bundled tests
        let tests = match pkg.tests() { Ok(t) => t, Err(e) => { Self::fail(&mut rep, "bundled_tests", e); return self.log_import(rep, path); } };
        if tests.len() != pkg.manifest.tests.count || tests.is_empty() {
            Self::fail(&mut rep, "bundled_tests", format!("expected {} tests, found {}", pkg.manifest.tests.count, tests.len())); return self.log_import(rep, path);
        }
        let mut correct = 0usize;
        for t in &tests {
            match model.run(&t.input) {
                Ok(out) if argmax(&out) == t.expected => correct += 1,
                Ok(_) => {}
                Err(e) => { Self::fail(&mut rep, "bundled_tests", e); return self.log_import(rep, path); }
            }
        }
        let acc = correct as f64 / tests.len() as f64;
        rep.test_accuracy = Some(acc);
        if acc < pkg.manifest.tests.min_accuracy {
            Self::fail(&mut rep, "bundled_tests", format!("accuracy {acc:.4} < required {:.4}", pkg.manifest.tests.min_accuracy));
            return self.log_import(rep, path);
        }
        Self::okstep(&mut rep, "bundled_tests", format!("accuracy {acc:.4} on {} cases", tests.len()));

        // resource check: probe latency; reject absurd per-inference cost
        let t0 = Instant::now();
        let _ = model.run(&tests[0].input);
        let us = t0.elapsed().as_micros();
        if us > 1_000_000 { Self::fail(&mut rep, "resource_check", format!("probe inference {us} us too slow")); return self.log_import(rep, path); }
        Self::okstep(&mut rep, "resource_check", format!("probe {us} us"));

        // activate: copy into store, update registry
        let hints = match pkg.hints() { Ok(h) => h, Err(e) => { Self::fail(&mut rep, "activate", e); return self.log_import(rep, path); } };
        let store_file = format!("{}.cap", pkg.manifest.package_id);
        if let Err(e) = std::fs::copy(path, self.root.join("store").join(&store_file)) { Self::fail(&mut rep, "activate", e); return self.log_import(rep, path); }
        let m = &pkg.manifest;
        let rec = CapabilityRecord {
            capability_id: m.capability_id.clone(), active_version: m.version.clone(), versions: vec![],
            description: hints.description, keywords: hints.keywords, input_dim: m.io.input.dim,
            labels: m.io.output.labels.clone(), device_caps: m.requires.device_caps.clone(),
            dependencies: m.dependencies.clone(), provenance: m.provenance.clone(), stats: Stats::default(),
        };
        let ver = VersionRecord { version: m.version.clone(), package_id: m.package_id.clone(), store_file,
            params: m.resources.params, model_bytes: m.resources.model_bytes, test_accuracy: acc, activated_at: now_secs() };
        if let Err(e) = self.registry.upsert(rec, ver) { Self::fail(&mut rep, "activate", e); return self.log_import(rep, path); }
        self.unload(&m.capability_id.clone()); // force reload of the new version
        Self::okstep(&mut rep, "activate", "registered");
        rep.activated = true;
        self.log_import(rep, path)
    }

    fn log_import(&self, rep: ImportReport, path: &Path) -> ImportReport {
        let _ = self.trace.append(&serde_json::json!({
            "event": "import", "ts": now_secs(), "file": path.file_name().map(|f| f.to_string_lossy().to_string()),
            "activated": rep.activated, "failed_step": rep.failed_step, "capability": rep.capability_id, "version": rep.version,
        }));
        rep
    }

    /// Load (and verify again) the active version of a capability from the store.
    fn ensure_loaded(&mut self, cap: &str) -> Result<()> {
        let rec = self.registry.get(cap).ok_or_else(|| anyhow!("unknown capability {cap}"))?.clone();
        let active = rec.active().clone();
        if let Some((v, _)) = self.loaded.get(cap) {
            if *v == active.version { self.touch(cap); return Ok(()); }
        }
        let pkg = CapPackage::open(&self.root.join("store").join(&active.store_file))?;
        pkg.verify_signature(&self.trust)?;
        pkg.verify_hashes()?;
        let model = self.backend.load(&pkg.manifest.model.format, pkg.model_bytes(), pkg.manifest.io.input.dim)?;
        while self.lru.len() >= self.device.max_active_modules.max(1) {
            let victim = self.lru.remove(0);
            self.loaded.remove(&victim);
        }
        self.loaded.insert(cap.to_string(), (active.version, model));
        self.touch(cap);
        Ok(())
    }

    fn touch(&mut self, cap: &str) {
        self.lru.retain(|c| c != cap);
        self.lru.push(cap.to_string());
    }

    pub fn solve(&mut self, task: &Task) -> Result<Outcome> {
        self.task_counter += 1;
        let task_id = format!("t{}-{}", now_secs(), self.task_counter);
        let decision = self.router.route(&self.registry, task);
        let out = match (&decision.status, &decision.chosen) {
            (Status::Unknown, _) | (_, None) => Outcome::NeedsHelp {
                reason: "no registered capability matches this task".into(), task_id: task_id.clone() },
            (status, Some(cap)) => {
                let cap = cap.clone();
                self.ensure_loaded(&cap)?;
                let mut ws = Workspace::new(&task.intent, &task.input);
                let t0 = Instant::now();
                let (ver, model) = self.loaded.get(&cap).unwrap();
                let logits = model.run(&task.input)?;
                let latency_us = t0.elapsed().as_micros() as u64;
                let ver = ver.clone();
                let probs = softmax(&logits);
                let idx = argmax(&probs);
                let confidence = probs[idx];
                let rec = self.registry.get(&cap).unwrap();
                let label = rec.labels.get(idx).cloned().unwrap_or_else(|| idx.to_string());
                let route_score = decision.candidates.iter().find(|c| c.capability_id == cap).map(|c| c.score).unwrap_or(0.0);
                let mut st = status.clone();
                if confidence < self.conf_threshold { st = Status::Uncertain; }
                ws.uncertainty = Some(1.0 - confidence);
                ws.history.push(Step { capability_id: cap.clone(), version: ver.clone(), output_label: label.clone(), probs: probs.clone(), latency_us });
                self.registry.record_call(&cap, latency_us)?;
                Outcome::Answer { status: st, capability_id: cap, version: ver, label, label_index: idx, probs, confidence, route_score, latency_us, task_id: task_id.clone() }
            }
        };
        self.trace.append(&serde_json::json!({
            "event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent,
            "routing": decision, "outcome": &out,
        }))?;
        Ok(out)
    }

    pub fn report_outcome(&mut self, cap: &str, success: bool) -> Result<()> { self.registry.record_outcome(cap, success) }

    /// Run bundled tests of the active version against the stored package (regression check).
    pub fn regression(&mut self, cap: &str) -> Result<f64> {
        let rec = self.registry.get(cap).ok_or_else(|| anyhow!("unknown capability {cap}"))?.clone();
        let pkg = CapPackage::open(&self.root.join("store").join(&rec.active().store_file))?;
        self.ensure_loaded(cap)?;
        let (_, model) = self.loaded.get(cap).unwrap();
        let tests = pkg.tests()?;
        let ok = tests.iter().filter(|t| model.run(&t.input).map(|o| argmax(&o) == t.expected).unwrap_or(false)).count();
        Ok(ok as f64 / tests.len() as f64)
    }

    pub fn manifest_of(&self, cap: &str) -> Result<Manifest> {
        let rec = self.registry.get(cap).ok_or_else(|| anyhow!("unknown capability {cap}"))?;
        Ok(CapPackage::open(&self.root.join("store").join(&rec.active().store_file))?.manifest)
    }
}

impl From<PackageError> for ImportReport {
    fn from(e: PackageError) -> Self {
        ImportReport { activated: false, package_id: None, capability_id: None, version: None,
            failed_step: Some("parse"), steps: vec![StepResult { step: "parse", ok: false, detail: e.to_string() }], test_accuracy: None }
    }
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}

pub fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&p), 2);
    }
}
