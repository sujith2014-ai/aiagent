"""Render benchmarks/reports/phase9.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase9.json").read_text())
nr = json.loads((ROOT / "benchmarks/reports/novelty_rules.json").read_text())
f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
L = [f"# Phase 9 benchmark summary ({r['date']}, {r['platform']})", "", f"Data: {r['datasets']}.", "",
     "Taught by example through the desktop app: **no teacher, no rule**. Accuracy is measured through the Rust runtime on the held-out 20% (refused rows count as misses in `overall`; `answered` excludes them). Intervals are 95% Wilson; test sets are small (30-360 rows).", "",
     "| task | n | gate | modular overall [95% CI] | coverage | answered acc | logistic reg. | kNN k=5 | sklearn MLP | majority | params (ours / LR / MLP) | model bytes | latency us p50 | ECE raw -> calibrated |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, e in r["tasks"].items():
    rt, b = e["runtime"], e["baselines"]
    L.append(f"| {k} | {e['data']['n']} | {e['promotion_gate']:.3f} | {rt['accuracy_overall']:.3f} {rt['ci95_overall']} | {rt['coverage']:.3f} | {f(rt['accuracy_answered'])} | {b['logistic_regression']['accuracy']:.3f} | {b['knn_k5']['accuracy']:.3f} | {b['sklearn_mlp_32x32']['accuracy']:.3f} | {b['majority']['accuracy']:.3f} | {rt['params']} / {b['logistic_regression']['params_or_stored_floats']} / {b['sklearn_mlp_32x32']['params_or_stored_floats']} | {rt['model_bytes']} | {rt['latency_us_p50']} | {f(rt['ece_raw_answered'])} -> {f(rt['ece_calibrated_answered'])} |")
L += ["", f"Max forgetting drop across the 4 sequential stages: {r['max_forgetting_drop']} (independent modules: zero by construction).", "", f"Routing by intent+input shape with all four installed (correct capability fraction): {r['routing_correct_capability_fraction']}.",
      f"Inputs shifted far out of range refused: {r['out_of_range_refused_fraction']}.", "", "## Novelty rule on the real tasks (Rust runtime)", "", "| task | strict: held-out coverage / +4sd shift refused | balanced: held-out coverage / +4sd shift refused |", "|---|---|---|"]
for k, v in r["novelty_rule_on_real_tasks"].items():
    L.append(f"| {k} | {v['strict']['heldout_coverage']:.3f} / {v['strict']['shift_4sd_refused']:.3f} | {v['balanced']['heldout_coverage']:.3f} / {v['balanced']['shift_4sd_refused']:.3f} |")
m = lambda rule, key: sum(x[rule][key] for x in r["novelty_rule_on_real_tasks"].values()) / len(r["novelty_rule_on_real_tasks"])
L += ["", f"Mean over tasks: strict coverage {m('strict', 'heldout_coverage'):.3f}, shift refused {m('strict', 'shift_4sd_refused'):.3f}; balanced coverage {m('balanced', 'heldout_coverage'):.3f}, shift refused {m('balanced', 'shift_4sd_refused'):.3f}.", "",
      "### Offline rule comparison (mean over the 4 datasets; benchmarks/reports/novelty_rules.json)", "", "| rule | false refusal on held-out | gross | +4 sd | -4 sd | single feature +25 sd | in-range uniform |", "|---|---|---|---|---|---|---|"]
for rule, v in nr["mean_over_datasets"].items():
    L.append(f"| {rule} | " + " | ".join(f"{x:.2f}" for x in v.values()) + " |")
rs = r["restart_and_constrained"]
L += ["", "## Clean restart and constrained device", "", f"Fresh runtime import ok: {rs['fresh_import_all_activated']}; constrained profile import ok: {rs['constrained_import_all_activated']}.", "", "| task | identical outputs after restart | constrained max prob diff | same refusals | constrained peak RSS KB |", "|---|---|---|---|---|"]
for k, v in rs["per_capability"].items():
    L.append(f"| {k} | {v['fresh_identical_to_before']} | {v['constrained_max_prob_diff']} | {v['same_refusals_constrained']} | {v['constrained_peak_rss_kb']} |")
(ROOT / "benchmarks/reports/phase9.md").write_text("\n".join(L)); print("ok")
