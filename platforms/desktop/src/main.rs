//! Desktop harness for the core. Thin by design: argument parsing, JSON output,
//! and PC-only resource measurement (peak RSS). No business logic.
use aicore::device::DeviceProfile;
use aicore::router::Task;
use aicore::runtime::{Outcome, Runtime};
use aicore::trust::TrustStore;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok())))
        .unwrap_or(0)
}

fn parse_input(s: &str) -> Result<Vec<f32>> {
    s.split(',').map(|x| x.trim().parse::<f32>().map_err(|e| anyhow!("bad number '{x}': {e}"))).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut root = PathBuf::from("./runtime-state");
    let mut trust = PathBuf::from("./trust.json");
    let mut device = "PC_FULL".to_string();
    let mut detect = "full".to_string();
    let mut force: Option<String> = None;
    let mut router_kind = "keyword".to_string();
    let mut no_stats = false;
    let mut novelty = "balanced".to_string();
    let mut backend = "auto".to_string();
    let mut rest = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => { root = args[i + 1].clone().into(); i += 2 }
            "--trust" => { trust = args[i + 1].clone().into(); i += 2 }
            "--device" => { device = args[i + 1].clone(); i += 2 }
            "--detect" => { detect = args[i + 1].clone(); i += 2 }
            "--capability" => { force = Some(args[i + 1].clone()); i += 2 }
            "--router" => { router_kind = args[i + 1].clone(); i += 2 }
            "--no-stats" => { no_stats = true; i += 1 }
            "--novelty" => { novelty = args[i + 1].clone(); i += 2 }
            "--backend" => { backend = args[i + 1].clone(); i += 2 }
            _ => { rest.push(args[i].clone()); i += 1 }
        }
    }
    if stateless(rest.first().map(|s| s.as_str()).unwrap_or(""), &rest)? { return Ok(()); }
    let dev = DeviceProfile::by_name(&device).ok_or_else(|| anyhow!("unknown device profile {device}"))?;
    let trust_store = TrustStore::from_file(&trust)?;
    let mut rt = Runtime::open(&root, dev, trust_store)?;
    rt.detect = aicore::runtime::Detection::by_name(&detect).ok_or_else(|| anyhow!("unknown --detect {detect}"))?;
    rt.set_backend(&backend)?;
    rt.record_stats = !no_stats;
    rt.detect.novelty_rule = aicore::runtime::NoveltyRule::by_name(&novelty).ok_or_else(|| anyhow!("unknown --novelty {novelty} (strict|balanced)"))?;
    rt.force_capability = force;
    if router_kind == "learned" { rt.use_learned_router()?; }
    let cmd = rest.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "import" => {
            let mut reports = vec![];
            for f in &rest[1..] { reports.push(serde_json::to_value(rt.import(&PathBuf::from(f)))?); }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        "archive" => { let c = rest.get(1).ok_or_else(|| anyhow!("capability id"))?; rt.archive(c)?; println!("{}", json!({"archived": c})); }
        "restore" => { let c = rest.get(1).ok_or_else(|| anyhow!("capability id"))?; rt.restore(c)?; println!("{}", json!({"restored": c})); }
        "list-all" => {
            let v: Vec<_> = rt.registry.all_including_hidden().map(|r| json!({"capability_id": r.capability_id, "role": r.role, "archived": r.archived,
                "active_version": r.active_version, "params": r.active().params, "model_bytes": r.active().model_bytes, "calls": r.stats.calls, "last_used": r.stats.last_used,
                "last_trained": r.stats.last_trained, "store_file": r.active().store_file})).collect();
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        "list" => {
            let v: Vec<_> = rt.registry.all().map(|r| json!({
                "capability_id": r.capability_id, "active_version": r.active_version, "store_file": r.active().store_file, "keywords": r.keywords,
                "versions": r.versions.iter().map(|v| &v.version).collect::<Vec<_>>(),
                "params": r.active().params, "model_bytes": r.active().model_bytes, "calls": r.stats.calls,
                "input_dim": r.input_dim, "labels": r.labels, "input_stats": r.input_stats, "dependencies": r.dependencies,
                "test_accuracy": r.active().test_accuracy, "stats": r.stats,
            })).collect();
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        "solve" => {
            let intent = flag(&rest, "--intent").ok_or_else(|| anyhow!("--intent required"))?;
            let input = parse_input(&flag(&rest, "--input").ok_or_else(|| anyhow!("--input required"))?)?;
            let out = rt.solve(&Task { intent, input })?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "regression" => {
            let cap = rest.get(1).ok_or_else(|| anyhow!("capability id required"))?;
            println!("{}", json!({"capability_id": cap, "bundled_test_accuracy": rt.regression(cap)?}));
        }
        "rollback" => {
            let cap = rest.get(1).ok_or_else(|| anyhow!("capability id required"))?;
            println!("{}", json!({"capability_id": cap, "active_version": rt.registry.rollback(cap)?}));
        }
        "reload-check" => {
            // solve -> unload -> solve again; outputs must be identical.
            let cap = rest.get(1).ok_or_else(|| anyhow!("capability id required"))?.clone();
            let intent = flag(&rest, "--intent").ok_or_else(|| anyhow!("--intent required"))?;
            let input = parse_input(&flag(&rest, "--input").ok_or_else(|| anyhow!("--input required"))?)?;
            let t = Task { intent, input };
            let a = serde_json::to_value(rt.solve(&t)?)?;
            let was_loaded = rt.unload(&cap);
            let still = rt.loaded_modules().contains(&cap);
            let b = serde_json::to_value(rt.solve(&t)?)?;
            let same = a["probs"] == b["probs"] && a["label"] == b["label"];
            println!("{}", json!({"was_loaded": was_loaded, "loaded_after_unload": still, "reloaded": rt.loaded_modules().contains(&cap), "identical": same}));
        }
        "plan-batch" => {
            // --plan FILE --cases FILE.jsonl ; case: {"inputs": {"slot": [..]}}. Each case runs once to warm, then is timed.
            let plan: aicore::graph::Plan = serde_json::from_slice(&std::fs::read(flag(&rest, "--plan").ok_or_else(|| anyhow!("--plan required"))?)?)?;
            let file = flag(&rest, "--cases").ok_or_else(|| anyhow!("--cases required"))?;
            let mut results = vec![];
            let (mut tot, mut modu, mut ovh, mut execs) = (vec![], vec![], vec![], vec![]);
            let mut needs_help = 0usize;
            for line in std::fs::read_to_string(file)?.lines().filter(|l| !l.trim().is_empty()) {
                let c: serde_json::Value = serde_json::from_str(line)?;
                let inputs: std::collections::BTreeMap<String, Vec<f32>> = c["inputs"].as_object().ok_or_else(|| anyhow!("inputs"))?.iter()
                    .map(|(k, v)| (k.clone(), v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())).collect();
                match rt.run_plan(&plan, &inputs) {
                    Ok(r) => { tot.push(r.total_us); modu.push(r.module_us); ovh.push(r.overhead_us); execs.push(r.node_executions);
                        results.push(json!({"outputs": r.outputs, "records": r.records.iter().map(|x| json!({"node": x.node, "op": x.op, "cap": x.capability, "label": x.label})).collect::<Vec<_>>(),
                                            "calls": r.capability_calls, "total_us": r.total_us, "module_us": r.module_us, "overhead_us": r.overhead_us, "levels": r.parallel_levels})); }
                    Err(aicore::graph::PlanError::NeedsHelp { node, reason }) => { needs_help += 1; results.push(json!({"needs_help": true, "node": node, "reason": reason})); }
                    Err(e) => { results.push(json!({"error": e.to_string()})); }
                }
            }
            let med = |mut v: Vec<u64>| { v.sort(); if v.is_empty() { 0 } else { v[v.len() / 2] } };
            println!("{}", serde_json::to_string(&json!({"summary": {"plan": plan.id, "cases": results.len(), "needs_help": needs_help,
                "total_us_p50": med(tot), "module_us_p50": med(modu), "overhead_us_p50": med(ovh), "node_executions_p50": med(execs.iter().map(|x| *x as u64).collect()),
                "peak_rss_kb": peak_rss_kb()}, "results": results}))?);
        }
        "plan-validate" => {
            let plan: aicore::graph::Plan = serde_json::from_slice(&std::fs::read(rest.get(1).ok_or_else(|| anyhow!("plan file"))?)?)?;
            match plan.validate() { Ok(()) => println!("{}", json!({"valid": true, "levels": plan.parallel_levels()})), Err(e) => println!("{}", json!({"valid": false, "error": e.to_string()})) }
        }
        "batch" => {
            // cases JSONL: {"intent": "...", "input": [..], "expected_index": n?, "expected_capability": "..."?}
            let file = flag(&rest, "--cases").ok_or_else(|| anyhow!("--cases required"))?;
            let rss_before = peak_rss_kb();
            let mut results = vec![];
            let (mut correct, mut scored, mut needs_help, mut lat) = (0usize, 0usize, 0usize, vec![]);
            let wall = Instant::now();
            for line in std::fs::read_to_string(file)?.lines().filter(|l| !l.trim().is_empty()) {
                let c: serde_json::Value = serde_json::from_str(line)?;
                let task = Task { intent: c["intent"].as_str().unwrap_or("").into(),
                    input: c["input"].as_array().ok_or_else(|| anyhow!("input"))?.iter().map(|x| x.as_f64().unwrap() as f32).collect() };
                let out = rt.solve(&task)?;
                match &out {
                    Outcome::Answer { capability_id, label_index, latency_us, probs, label, status, confidence, raw_confidence, flags, .. } => {
                        lat.push(*latency_us);
                        let cap_ok = c.get("expected_capability").and_then(|v| v.as_str()).map(|e| e == capability_id);
                        let idx_ok = c.get("expected_index").and_then(|v| v.as_u64()).map(|e| e as usize == *label_index);
                        if let Some(ok) = idx_ok { scored += 1; if ok { correct += 1 }; rt.report_outcome(capability_id, ok)?; }
                        results.push(json!({"capability": capability_id, "label": label, "label_index": label_index, "status": status, "confidence": confidence, "raw_confidence": raw_confidence, "flags": flags, "probs": probs, "latency_us": latency_us, "index_ok": idx_ok, "capability_ok": cap_ok}));
                    }
                    Outcome::NeedsHelp { reason, reason_code, .. } => { needs_help += 1; results.push(json!({"needs_help": true, "reason": reason, "reason_code": reason_code})); }
                }
            }
            lat.sort();
            let pct = |p: f64| if lat.is_empty() { 0 } else { lat[((lat.len() as f64 - 1.0) * p) as usize] };
            let summary = json!({
                "device": rt.device.name, "cases": results.len(), "scored": scored,
                "accuracy": if scored > 0 { correct as f64 / scored as f64 } else { f64::NAN },
                "needs_help": needs_help, "latency_us_p50": pct(0.5), "latency_us_p95": pct(0.95),
                "wall_ms": wall.elapsed().as_millis() as u64,
                "peak_rss_kb": peak_rss_kb(), "peak_rss_kb_before": rss_before,
                "active_params": rt.registry.all().map(|r| r.active().params).sum::<u64>(),
                "package_bytes": rt.registry.all().map(|r| r.active().model_bytes).sum::<u64>(),
                "loaded_modules": rt.loaded_modules(),
            });
            println!("{}", serde_json::to_string_pretty(&json!({"summary": summary, "results": results}))?);
        }
        _ => eprintln!("commands: import <cap...> | list | solve --intent S --input a,b | regression CAP | rollback CAP | reload-check CAP --intent S --input a,b | batch --cases FILE\nglobal: --root DIR --trust FILE --device PC_FULL|PC_CONSTRAINED"),
    }
    Ok(())
}


/// Commands that need no runtime state (no trust store, no registry): they must work with nothing but their arguments.
fn stateless(cmd: &str, rest: &[String]) -> Result<bool> {
    let rest = rest.to_vec();
    match cmd {
        "policy-check" => {
            // --policy FILE --request JSON [--approvals FILE] ; prints the decision as JSON (exit 0 always; the effect is in the output)
            let policy: aicore::policy::Policy = serde_json::from_slice(&std::fs::read(flag(&rest, "--policy").ok_or_else(|| anyhow!("--policy"))?)?)?;
            let req: aicore::policy::Request = serde_json::from_str(&flag(&rest, "--request").ok_or_else(|| anyhow!("--request"))?)?;
            let approvals: Vec<aicore::policy::Approval> = match flag(&rest, "--approvals") { Some(f) if std::path::Path::new(&f).exists() => serde_json::from_slice(&std::fs::read(f)?)?, _ => vec![] };
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            let mut d = serde_json::to_value(aicore::policy::evaluate(&policy, &req, &approvals, now))?;
            d["request_hash"] = json!(aicore::policy::request_hash(&req));
            println!("{}", d);
        }
        "approve" => {
            // operator-only: --approvals FILE --request JSON --approver NAME --ttl SECONDS
            let req: aicore::policy::Request = serde_json::from_str(&flag(&rest, "--request").ok_or_else(|| anyhow!("--request"))?)?;
            let file = flag(&rest, "--approvals").ok_or_else(|| anyhow!("--approvals"))?;
            let ttl: u64 = flag(&rest, "--ttl").and_then(|t| t.parse().ok()).unwrap_or(300);
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            let mut list: Vec<aicore::policy::Approval> = if std::path::Path::new(&file).exists() { serde_json::from_slice(&std::fs::read(&file)?)? } else { vec![] };
            list.push(aicore::policy::Approval { request_hash: aicore::policy::request_hash(&req), approver: flag(&rest, "--approver").unwrap_or_else(|| "operator".into()), approved_at: now, expires_at: now + ttl });
            std::fs::write(&file, serde_json::to_vec_pretty(&list)?)?;
            println!("{}", json!({"approved": aicore::policy::request_hash(&req), "expires_at": now + ttl}));
        }
        "features" => {
            let dim: usize = flag(&rest, "--dim").and_then(|d| d.parse().ok()).unwrap_or(256);
            let text = flag(&rest, "--text").ok_or_else(|| anyhow!("--text"))?;
            println!("{}", serde_json::to_string(&aicore::runtime::Runtime::features(&text, dim))?);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
