"""Render benchmarks/reports/phase6.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase6.json").read_text())
f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
L = [f"# Phase 6 benchmark summary ({r['date']}, {r['platform']})", "", "## Teacher rule fault matrix (spec mode; environment-verified at 0.95)", "", "| fault | compare_numbers | point_region | argmax_position |", "|---|---|---|---|"]
for k, d in r["teacher_fault_matrix"].items():
    L.append(f"| {k} | " + " | ".join(d[t]["result"] for t in ("compare_numbers", "point_region", "argmax_position")) + " |")
L += ["", "`subtle_wrong` is only subtle for compare_numbers (wrong on ~1% of inputs, accepted); the other two use the gross faults.", "", "## Provider path (mock HTTP endpoints)", ""]
for k, v in r["http_providers"].items():
    L.append(f"- {k}: {v if not isinstance(v, dict) else {a: b for a, b in v.items()}}")
p = r["paraphrase_teacher_dependency"]
L += ["", "## Paraphrase handling: router_update vs per-task rerouting (36 paraphrase tasks, 3 rounds)", "", "| mode | teacher calls | per round | accepted router updates | answered |", "|---|---|---|---|---|"]
for k, v in p.items():
    L.append(f"| {k} | {v['teacher_calls_after_learning']} | {v['per_round_teacher_calls']} | {v['router_updates_accepted']} | {v['answered']}/{v['tasks']} |")
d = r["domain_extension_compare_wide"]
L += ["", "## Domain extension (compare_numbers must work on a 4x wider range)", "", f"Before: `{d['before_extension_outcome']}`. Chosen strategy: **{d['decision']['chosen']}**.", "",
      "| strategy | new-domain verification | old-domain before -> after | params added | passed |", "|---|---|---|---|---|"]
for a in d["decision"]["attempts"]:
    L.append(f"| {a['strategy']} | {f(a.get('new_acc_verification'))} | {f(a.get('old_acc_before', ''))} -> {f(a.get('old_acc_after', ''))} | {a.get('params_added', '')} | {a['passed']} |")
n = d["naive_finetune_no_replay_ablation"]
L += ["", f"Ablation (not installed): naive fine-tune without replay: new {f(n['new_acc_verification'])}, old {f(n['old_acc_before'])} -> {f(n['old_acc_after'])} ({n['reason']}).", f"After extension: {d['after_extension']}; bundled regression {d['bundled_regression_after']['bundled_test_accuracy']}.", ""]
c = r["conflicting_rule_change"]
L += ["## Conflicting rule change (point_region now means a smaller circle)", "", f"Chosen strategy: **{c['decision']['chosen']}**.", "", "| strategy | new-domain verification | old-domain before -> after | params added | passed |", "|---|---|---|---|---|"]
for a in c["decision"]["attempts"]:
    L.append(f"| {a['strategy']} | {f(a.get('new_acc_verification'))} | {f(a.get('old_acc_before', ''))} -> {f(a.get('old_acc_after', ''))} | {a.get('params_added', '')} | {a['passed']} |")
L += ["", f"Routing afterwards: {c['routing_after']}. Probe: {c['probe']}", ""]
(ROOT / "benchmarks/reports/phase6.md").write_text("\n".join(L)); print("ok")
