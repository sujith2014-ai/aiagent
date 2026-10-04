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
    let mut rest = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => { root = args[i + 1].clone().into(); i += 2 }
            "--trust" => { trust = args[i + 1].clone().into(); i += 2 }
            "--device" => { device = args[i + 1].clone(); i += 2 }
            _ => { rest.push(args[i].clone()); i += 1 }
        }
    }
    let dev = DeviceProfile::by_name(&device).ok_or_else(|| anyhow!("unknown device profile {device}"))?;
    let trust_store = TrustStore::from_file(&trust)?;
    let mut rt = Runtime::open(&root, dev, trust_store)?;
    let cmd = rest.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "import" => {
            let mut reports = vec![];
            for f in &rest[1..] { reports.push(serde_json::to_value(rt.import(&PathBuf::from(f)))?); }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        "list" => {
            let v: Vec<_> = rt.registry.all().map(|r| json!({
                "capability_id": r.capability_id, "active_version": r.active_version,
                "versions": r.versions.iter().map(|v| &v.version).collect::<Vec<_>>(),
                "params": r.active().params, "model_bytes": r.active().model_bytes,
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
                    Outcome::Answer { capability_id, label_index, latency_us, probs, label, status, .. } => {
                        lat.push(*latency_us);
                        let cap_ok = c.get("expected_capability").and_then(|v| v.as_str()).map(|e| e == capability_id);
                        let idx_ok = c.get("expected_index").and_then(|v| v.as_u64()).map(|e| e as usize == *label_index);
                        if let Some(ok) = idx_ok { scored += 1; if ok { correct += 1 }; rt.report_outcome(capability_id, ok)?; }
                        results.push(json!({"capability": capability_id, "label": label, "status": status, "probs": probs, "latency_us": latency_us, "index_ok": idx_ok, "capability_ok": cap_ok}));
                    }
                    Outcome::NeedsHelp { reason, .. } => { needs_help += 1; results.push(json!({"needs_help": true, "reason": reason})); }
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

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
