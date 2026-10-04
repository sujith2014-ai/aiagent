//! Runtime: guarded import (the activation sequence), lazy load/unload, solve.
use crate::backend::{default_backend, Backend, Model};
use crate::capability::Manifest;
use crate::device::DeviceProfile;
use crate::graph::{NodeRecord, Node, Plan, PlanError, PlanResult, Ref, MAX_STEPS};
use crate::package::{CapPackage, PackageError};
use crate::registry::{now_secs, CapabilityRecord, Registry, Stats, VersionRecord};
use crate::router::{hash_features, KeywordRouter, LearnedRouter, Router, Status, Task};
use crate::trace::Trace;
use crate::trust::TrustStore;
use crate::workspace::{Step, Workspace};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
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
        label: String, label_index: usize, probs: Vec<f32>, confidence: f32, raw_confidence: f32,
        route_score: f64, novelty: Option<Novelty>, flags: Vec<&'static str>, latency_us: u64, task_id: String,
    },
    #[serde(rename = "NEEDS_HELP")]
    NeedsHelp { reason_code: &'static str, reason: String, task_id: String },
}

/// When an input counts as outside the learned domain, from the statistics shipped in the package.
/// strict: any feature outside range+10%. balanced (default): >=2 features outside range+10%, or any |z|>12 (std floored at 5% of range).
/// Measured trade-off on 4 real datasets (benchmarks/reports/novelty_rules.json): false refusals on held-out data 3% -> 1%, detection of a
/// +-4 sd shift on every feature 97% -> 93%, gross/single-extreme-feature shifts still ~100%.
#[derive(Clone, Debug, Serialize)]
pub struct NoveltyRule { pub min_count: usize, pub max_z: f32, pub std_floor_frac: f32, pub margin_frac: f32 }

impl NoveltyRule {
    pub fn strict() -> Self { Self { min_count: 1, max_z: f32::INFINITY, std_floor_frac: 0.05, margin_frac: 0.1 } }
    pub fn balanced() -> Self { Self { min_count: 2, max_z: 12.0, std_floor_frac: 0.05, margin_frac: 0.1 } }
    pub fn by_name(n: &str) -> Option<Self> { match n { "strict" => Some(Self::strict()), "balanced" => Some(Self::balanced()), _ => None } }
    pub fn flags(&self, n: &Novelty) -> bool { n.out_of_range_count >= self.min_count || n.max_z > self.max_z }
}

/// Which unknown/novelty signals are active (ablatable in experiments).
#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub novelty_rule: NoveltyRule,
    pub use_novelty: bool,
    pub use_calibration: bool,
    pub use_confidence: bool,
    pub use_history: bool,
    pub conf_threshold: f32,
}

impl Detection {
    pub fn by_name(n: &str) -> Option<Self> {
        let (nov, cal, conf, hist) = match n {
            "keyword" => (false, false, false, false),
            "novelty" => (true, false, false, false),
            "confidence" => (true, false, true, false),   // raw (uncalibrated) softmax confidence
            "calibrated" => (true, true, true, false),
            "full" => (true, true, true, true),
            _ => return None,
        };
        Some(Self { novelty_rule: NoveltyRule::balanced(), use_novelty: nov, use_calibration: cal, use_confidence: conf, use_history: hist, conf_threshold: 0.7 })
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct Novelty { pub out_of_range_frac: f32, pub out_of_range_count: usize, pub max_z: f32 }

fn novelty_of(rec: &CapabilityRecord, x: &[f32], rule: &NoveltyRule) -> Option<Novelty> {
    let st = rec.input_stats.as_ref()?;
    if st.min.len() != x.len() { return None; }
    let mut oor = 0usize;
    let mut max_z = 0f32;
    for i in 0..x.len() {
        let range = (st.max[i] - st.min[i]).abs().max(1e-6);
        if x[i] < st.min[i] - rule.margin_frac * range || x[i] > st.max[i] + rule.margin_frac * range { oor += 1; }
        max_z = max_z.max((x[i] - st.mean[i]).abs() / st.std[i].max(rule.std_floor_frac * range).max(1e-6));
    }
    Some(Novelty { out_of_range_frac: oor as f32 / x.len() as f32, out_of_range_count: oor, max_z })
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
    pub detect: Detection,
    /// When set, the router is bypassed and this capability handles every task (used for verification).
    pub force_capability: Option<String>,
    /// When false, usage statistics are not recorded (admin probes / verification must not look like real usage).
    pub record_stats: bool,
}

impl Runtime {
    pub fn open(root: &Path, device: DeviceProfile, trust: TrustStore) -> Result<Self> {
        std::fs::create_dir_all(root.join("store"))?;
        Ok(Self {
            root: root.to_path_buf(), device, trust,
            registry: Registry::open(root)?,
            backend: default_backend("auto")?,
            router: Box::new(KeywordRouter::default()),
            loaded: HashMap::new(), lru: vec![],
            trace: Trace::open(root), task_counter: 0, detect: Detection::by_name("full").unwrap(), force_capability: None, record_stats: true,
        })
    }

    /// Select the inference backend: "auto" (mlp-lite, then the general runtime if compiled in), "mlp", or "tract". Unloads all modules.
    pub fn set_backend(&mut self, kind: &str) -> Result<()> {
        self.backend = default_backend(kind)?; self.loaded.clear(); self.lru.clear(); Ok(())
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
        Self::okstep(&mut rep, "sandbox_load", model.backend());

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
            input_stats: m.input_stats.clone(), calibration: m.calibration.clone(),
            role: m.role.clone().unwrap_or_else(|| "capability".into()), archived: false,
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
        if !task.input.iter().all(|v| v.is_finite()) {
            // NaN/inf compare false against every threshold, so they would slip past all guards and produce a garbage answer
            let o = Outcome::NeedsHelp { reason_code: "INVALID_INPUT", reason: "input contains NaN or infinite values".into(), task_id: task_id.clone() };
            self.trace.append(&serde_json::json!({"event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent, "outcome": &o}))?;
            return Ok(o);
        }
        let decision = match &self.force_capability {
            Some(c) => crate::router::Decision { status: Status::Known, chosen: Some(c.clone()),
                candidates: vec![crate::router::Candidate { capability_id: c.clone(), score: 1.0, dice: 1.0, shape_ok: true }] },
            None => self.router.route(&self.registry, task),
        };
        let out = match (&decision.status, &decision.chosen) {
            (Status::Unknown, _) | (_, None) => {
                let shape_any = decision.candidates.iter().any(|c| c.shape_ok);
                Outcome::NeedsHelp { reason_code: if shape_any { "NO_MATCH" } else { "SHAPE_OR_EMPTY" },
                    reason: "no registered capability matches this task".into(), task_id: task_id.clone() }
            }
            (status, Some(cap)) => {
                let cap = cap.clone();
                if let Err(e) = self.ensure_loaded(&cap) {
                    // the stored package no longer verifies (tampered store, revoked signer, ...): refuse cleanly, never answer from an unverified model
                    let o = Outcome::NeedsHelp { reason_code: "MODEL_UNAVAILABLE", reason: format!("capability '{cap}' cannot be loaded: {e}"), task_id: task_id.clone() };
                    self.trace.append(&serde_json::json!({"event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent, "routing": decision, "outcome": &o}))?;
                    return Ok(o);
                }
                let mut ws = Workspace::new(&task.intent, &task.input);
                let t0 = Instant::now();
                let (ver, model) = self.loaded.get(&cap).unwrap();
                let logits = model.run(&task.input)?;
                let latency_us = t0.elapsed().as_micros() as u64;
                let ver = ver.clone();
                if !logits.iter().all(|v| v.is_finite()) {
                    let o = Outcome::NeedsHelp { reason_code: "NON_FINITE_OUTPUT", reason: format!("module '{cap}' produced non-finite values for this input"), task_id: task_id.clone() };
                    self.trace.append(&serde_json::json!({"event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent, "routing": decision, "outcome": &o}))?;
                    return Ok(o);
                }
                let raw_probs = softmax(&logits);
                let rec = self.registry.get(&cap).unwrap();
                let temp = if self.detect.use_calibration { rec.calibration.as_ref().map(|c| c.temperature).unwrap_or(1.0) } else { 1.0 };
                let probs = if temp != 1.0 { softmax(&logits.iter().map(|l| l / temp).collect::<Vec<_>>()) } else { raw_probs.clone() };
                let idx = argmax(&probs);
                let confidence = probs[idx];
                let raw_confidence = raw_probs[argmax(&raw_probs)];
                let label = rec.labels.get(idx).cloned().unwrap_or_else(|| idx.to_string());
                let route_score = decision.candidates.iter().find(|c| c.capability_id == cap).map(|c| c.score).unwrap_or(0.0);
                let novelty = novelty_of(rec, &task.input, &self.detect.novelty_rule);
                let mut flags: Vec<&'static str> = vec![];
                let mut st = status.clone();
                if let (true, Some(n)) = (self.detect.use_novelty, &novelty) {
                    if self.detect.novelty_rule.flags(n) { flags.push("OUT_OF_DISTRIBUTION"); }
                }
                if self.detect.use_confidence && confidence < self.detect.conf_threshold { flags.push("LOW_CONFIDENCE"); st = Status::Uncertain; }
                if self.detect.use_history {
                    let (s_, f_) = (rec.stats.success, rec.stats.failure);
                    if s_ + f_ >= 20 && (s_ as f64) / ((s_ + f_) as f64) < 0.5 { flags.push("POOR_HISTORY"); st = Status::Uncertain; }
                }
                if flags.contains(&"OUT_OF_DISTRIBUTION") {
                    // out-of-domain input: refuse rather than extrapolate
                    let o = Outcome::NeedsHelp { reason_code: "OUT_OF_DISTRIBUTION", reason: format!("input outside the learned domain of '{cap}'"), task_id: task_id.clone() };
                    self.trace.append(&serde_json::json!({"event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent, "routing": decision, "outcome": &o}))?;
                    return Ok(o);
                }
                ws.uncertainty = Some(1.0 - confidence);
                ws.history.push(Step { capability_id: cap.clone(), version: ver.clone(), output_label: label.clone(), probs: probs.clone(), latency_us });
                if self.record_stats { self.registry.record_call(&cap, latency_us)?; }
                Outcome::Answer { status: st, capability_id: cap, version: ver, label, label_index: idx, probs, confidence, raw_confidence, route_score, novelty, flags, latency_us, task_id: task_id.clone() }
            }
        };
        self.trace.append(&serde_json::json!({
            "event": "task", "task_id": task_id, "ts": now_secs(), "intent": task.intent,
            "routing": decision, "outcome": &out,
        }))?;
        Ok(out)
    }

    /// Execute a validated plan. Capability nodes resolve by id or by intent through the router
    /// (intent resolution must be KNOWN; otherwise the whole plan stops with NEEDS_HELP).
    pub fn run_plan(&mut self, plan: &Plan, inputs: &BTreeMap<String, Vec<f32>>) -> Result<PlanResult, PlanError> {
        plan.validate()?;
        let mut slots: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (name, dim) in &plan.inputs {
            let v = inputs.get(name).ok_or_else(|| PlanError::Invalid(format!("missing input '{name}'")))?;
            if v.len() != *dim { return Err(PlanError::Invalid(format!("input '{name}' has {} values, plan declares {dim}", v.len()))); }
            if !v.iter().all(|x| x.is_finite()) { return Err(PlanError::Invalid(format!("input '{name}' contains NaN or infinite values"))); }
            slots.insert(name.clone(), v.clone());
        }
        let t0 = Instant::now();
        let mut st = ExecState { records: vec![], steps: 0, module_us: 0, calls: BTreeMap::new() };
        self.exec_nodes(&plan.nodes, &mut slots, &mut st)?;
        let outputs = resolve_all(&plan.outputs, &slots).map_err(|e| PlanError::Exec { node: "outputs".into(), reason: e })?;
        let total_us = t0.elapsed().as_micros() as u64;
        let res = PlanResult { plan_id: plan.id.clone(), outputs, node_executions: st.steps, total_us, module_us: st.module_us,
            overhead_us: total_us.saturating_sub(st.module_us), parallel_levels: plan.parallel_levels(),
            capability_calls: st.calls, records: st.records };
        let _ = self.trace.append(&serde_json::json!({"event": "plan", "ts": now_secs(), "plan": plan.id,
            "records": res.records, "total_us": total_us, "module_us": res.module_us}));
        Ok(res)
    }

    fn exec_nodes(&mut self, nodes: &[Node], slots: &mut BTreeMap<String, Vec<f32>>, st: &mut ExecState) -> Result<(), PlanError> {
        for n in nodes {
            st.steps += 1;
            if st.steps > MAX_STEPS { return Err(PlanError::Budget(MAX_STEPS)); }
            let id = n.id().to_string();
            let ex = |e: String| PlanError::Exec { node: id.clone(), reason: e };
            match n {
                Node::Gather { args, out, .. } => { let v = resolve_all(args, slots).map_err(ex)?; slots.insert(out.clone(), v); st.rec(&id, "gather", None, None, None, 0); }
                Node::Affine { args, scale, offset, out, .. } => {
                    let s: f32 = resolve_all(args, slots).map_err(ex)?.iter().sum();
                    slots.insert(out.clone(), vec![scale * s + offset]); st.rec(&id, "affine", None, None, None, 0);
                }
                Node::Select { index, options, out, .. } => {
                    let i = resolve_all(std::slice::from_ref(index), slots).map_err(ex)?;
                    let i = *i.first().ok_or_else(|| PlanError::Exec { node: id.clone(), reason: "empty index".into() })? as usize;
                    let opt = options.get(i).ok_or_else(|| PlanError::Exec { node: id.clone(), reason: format!("index {i} out of range") })?;
                    let v = resolve_all(std::slice::from_ref(opt), slots).map_err(|e| PlanError::Exec { node: id.clone(), reason: e })?;
                    slots.insert(out.clone(), v); st.rec(&id, "select", None, None, Some(i), 0);
                }
                Node::CondSwap { pair, flag, swap_if, out, .. } => {
                    let f = resolve_all(std::slice::from_ref(flag), slots).map_err(ex)?[0] as usize;
                    let a = resolve_all(&pair[..1], slots).map_err(|e| PlanError::Exec { node: id.clone(), reason: e })?;
                    let b = resolve_all(&pair[1..], slots).map_err(|e| PlanError::Exec { node: id.clone(), reason: e })?;
                    let v = if swap_if.contains(&f) { [b, a].concat() } else { [a, b].concat() };
                    slots.insert(out.clone(), v); st.rec(&id, "cond_swap", None, None, Some(f), 0);
                }
                Node::If { cond, in_set, then, otherwise, .. } => {
                    let c = resolve_all(std::slice::from_ref(cond), slots).map_err(ex)?[0] as usize;
                    let take = in_set.contains(&c);
                    st.rec(&id, if take { "if:then" } else { "if:else" }, None, None, Some(c), 0);
                    self.exec_nodes(if take { then } else { otherwise }, slots, st)?;
                }
                Node::Repeat { times, body, .. } => {
                    for _ in 0..*times { self.exec_nodes(body, slots, st)?; }
                }
                Node::Cap { capability, intent, args, out, .. } => {
                    let input = resolve_all(args, slots).map_err(ex)?;
                    let cap = match (capability, intent) {
                        (Some(c), _) => c.clone(),
                        (None, Some(i)) => {
                            let d = self.router.route(&self.registry, &Task { intent: i.clone(), input: input.clone() });
                            match (d.status, d.chosen) {
                                (Status::Known, Some(c)) => c,
                                (s, _) => return Err(PlanError::NeedsHelp { node: id.clone(), reason: format!("intent '{i}' resolved as {s:?}") }),
                            }
                        }
                        _ => unreachable!("validated"),
                    };
                    if !input.iter().all(|v| v.is_finite()) {
                        return Err(PlanError::NeedsHelp { node: id.clone(), reason: "INVALID_INPUT: non-finite value reached a capability".into() });
                    }
                    if self.detect.use_novelty {
                        // the same out-of-domain refusal as `solve`: a plan must not push inputs through a capability outside its learned domain
                        if let Some(rec) = self.registry.get(&cap) {
                            if let Some(n) = novelty_of(rec, &input, &self.detect.novelty_rule) {
                                if self.detect.novelty_rule.flags(&n) {
                                    return Err(PlanError::NeedsHelp { node: id.clone(), reason: format!("OUT_OF_DISTRIBUTION: input outside the learned domain of '{cap}'") });
                                }
                            }
                        }
                    }
                    self.ensure_loaded(&cap).map_err(|e| PlanError::Exec { node: id.clone(), reason: e.to_string() })?;
                    let (ver, model) = self.loaded.get(&cap).unwrap();
                    let ver = ver.clone();
                    let t = Instant::now();
                    let logits = model.run(&input).map_err(|e| PlanError::Exec { node: id.clone(), reason: e.to_string() })?;
                    let us = t.elapsed().as_micros() as u64;
                    if !logits.iter().all(|v| v.is_finite()) { return Err(PlanError::NeedsHelp { node: id.clone(), reason: format!("NON_FINITE_OUTPUT from '{cap}'") }); }
                    st.module_us += us;
                    let probs = softmax(&logits);
                    let idx = argmax(&probs);
                    slots.insert(out.clone(), vec![idx as f32]);
                    slots.insert(format!("{out}.conf"), vec![probs[idx]]);
                    *st.calls.entry(cap.clone()).or_insert(0) += 1;
                    st.rec(&id, "cap", Some(cap.clone()), Some(ver), Some(idx), us);
                    if self.record_stats { let _ = self.registry.record_call(&cap, us); }
                }
            }
        }
        Ok(())
    }

    /// Switch to the learned router stored in the registry (role "router"). Fails if none is installed.
    pub fn use_learned_router(&mut self) -> Result<()> {
        let rec = self.registry.router_record().ok_or_else(|| anyhow!("no learned router installed"))?.clone();
        let pkg = CapPackage::open(&self.root.join("store").join(&rec.active().store_file))?;
        pkg.verify_signature(&self.trust)?;
        pkg.verify_hashes()?;
        let model = self.backend.load(&pkg.manifest.model.format, pkg.model_bytes(), pkg.manifest.io.input.dim)?;
        self.router = Box::new(LearnedRouter { model, classes: pkg.manifest.io.output.labels.clone(), dim: pkg.manifest.io.input.dim, known_p: 0.80, uncertain_p: 0.50 });
        Ok(())
    }

    pub fn use_keyword_router(&mut self) { self.router = Box::new(KeywordRouter::default()); }

    /// Hide a capability from routing without deleting anything. Refused if an active capability depends on it.
    pub fn archive(&mut self, cap: &str) -> Result<()> {
        let dependents: Vec<String> = self.registry.all().filter(|r| r.capability_id != cap && r.dependencies.iter().any(|d| d == cap)).map(|r| r.capability_id.clone()).collect();
        if !dependents.is_empty() { return Err(anyhow!("cannot archive '{cap}': required by {dependents:?}")); }
        self.registry.set_archived(cap, true)?;
        self.unload(cap);
        Ok(())
    }

    pub fn restore(&mut self, cap: &str) -> Result<()> { self.registry.set_archived(cap, false) }

    pub fn features(text: &str, dim: usize) -> Vec<f32> { hash_features(text, dim) }

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

struct ExecState { records: Vec<NodeRecord>, steps: usize, module_us: u64, calls: BTreeMap<String, usize> }
impl ExecState {
    fn rec(&mut self, node: &str, op: &'static str, capability: Option<String>, version: Option<String>, label: Option<usize>, latency_us: u64) {
        self.records.push(NodeRecord { node: node.into(), op, capability, version, label, latency_us });
    }
}

fn resolve(r: &Ref, slots: &BTreeMap<String, Vec<f32>>) -> Result<Vec<f32>, String> {
    match r {
        Ref::Const { value } => Ok(vec![*value]),
        Ref::Slot { slot, index } => {
            let v = slots.get(slot).ok_or_else(|| format!("slot '{slot}' not set"))?;
            match index { None => Ok(v.clone()), Some(i) => v.get(*i).map(|x| vec![*x]).ok_or_else(|| format!("slot '{slot}' has no index {i}")) }
        }
    }
}

fn resolve_all(rs: &[Ref], slots: &BTreeMap<String, Vec<f32>>) -> Result<Vec<f32>, String> {
    let mut out = vec![];
    for r in rs { out.extend(resolve(r, slots)?); }
    Ok(out)
}

#[cfg(test)]
mod novelty_tests {
    use super::*;
    fn n(count: usize, z: f32) -> Novelty { Novelty { out_of_range_frac: 0.0, out_of_range_count: count, max_z: z } }
    #[test]
    fn strict_flags_any_single_feature_outside_range() {
        let r = NoveltyRule::strict();
        assert!(!r.flags(&n(0, 3.0)) && r.flags(&n(1, 3.0)) && r.flags(&n(5, 3.0)));
    }
    #[test]
    fn balanced_needs_two_features_or_an_extreme_z() {
        let r = NoveltyRule::balanced();
        assert!(!r.flags(&n(0, 3.0)) && !r.flags(&n(1, 3.8)));          // one moderately out-of-range feature is tolerated
        assert!(r.flags(&n(2, 3.0)) && r.flags(&n(1, 12.5)) && r.flags(&n(0, 30.0)));
    }
    #[test]
    fn rules_are_selectable_by_name() {
        assert!(NoveltyRule::by_name("strict").is_some() && NoveltyRule::by_name("balanced").is_some() && NoveltyRule::by_name("x").is_none());
    }
}
