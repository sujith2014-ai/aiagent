"""Render benchmarks/reports/phase12.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase12.json").read_text())
L = [f"# Phase 12 benchmark summary ({r['date']}, {r['platform']})", "", f"On-device runs: {r['constraint']}. **Not measured:** {', '.join(r['not_measured'])}.", "",
     "## Training from scratch: device (Rust trainer) vs server (PyTorch build service), 5 random 70/30 splits per task", "",
     "| task | train / eval rows | device acc (mean±sd) | server acc (mean±sd) | device train s | device wall s* | device peak RSS MB | device pkg bytes | server train+sign s | server pkg bytes |", "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in r["from_scratch"].items():
    d, s = v["device"], v["server"]
    L.append(f"| {k} | {v['train_examples']} / {v['eval_examples']} | {d['accuracy_mean']:.3f}±{d['accuracy_std']:.3f} | {s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f} | {d['train_s_mean']:.3f} | {d['wall_s_mean_incl_process_and_pack']:.3f} | {d['peak_rss_mb_max']} | {d['package_bytes_mean']:.0f} | {s['train_and_sign_s_mean']:.2f} | {s['package_bytes_mean']:.0f} |")
L += ["", "*wall includes spawning the CLI process, building, signing and importing the package.", f"Server process peak RSS (whole harness, includes PyTorch and scikit-learn): {r['server_process_peak_rss_mb']} MB.", "",
      "## Few-shot adaptation to an input drift (two highest-variance features read 1.5 sd high), 5 draws", "",
      "`frozen` = the unchanged model. `head`/`full` = on-device adaptation with the acceptance gates disabled (raw effect). `new_small_module` = train a fresh module on only the new samples. `server_retrain` = PyTorch on old + new data (the server holds the data; the device does not).", "",
      "'gate acceptance' = share of draws in which the default gates (better on new data, old-domain accuracy not down by more than 0.05) would have accepted the update.", ""]
for k, v in r["adaptation"].items():
    L += [f"### {k} (drifted feature columns {v['drift_columns']})", "", "| n adaptation samples | method | new-domain acc | old-domain acc | train s | gate acceptance |", "|---|---|---|---|---|---|"]
    for n, ms in v["n"].items():
        for m, x in ms.items():
            L.append(f"| {n} | {m} | {x['new_domain_accuracy_mean']:.3f}±{x['new_std']:.3f} | {x['old_domain_accuracy_mean']:.3f} | {x['train_s_mean']:.3f} | {x['default_gate_acceptance']} |")
    refs = ", ".join(f"n={n}: {f['count']}/5 refused ({f['reason']})" for n, f in v["new_module_refusals"].items() if f["count"])
    srv = ", ".join(f"n={n}: {f['count']}/5 failed ({f['reason']})" for n, f in v.get("server_retrain_failures", {}).items() if f["count"])
    L += ["", f"New small module: {refs or 'not refused'}." + (f" Server retrain: {srv}." if srv else ""), ""]
(ROOT / "benchmarks/reports/phase12.md").write_text("\n".join(L)); print("ok")
