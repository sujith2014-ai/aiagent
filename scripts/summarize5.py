"""Render benchmarks/reports/phase5.json as markdown (numbers only from the JSON)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase5.json").read_text())
L = [f"# Phase 5 benchmark summary ({r['date']}, {r['platform']})", "", "## Calibration (temperature scaling fit on validation, ECE on held-out)", "", "| capability | T | ECE before | ECE after |", "|---|---|---|---|"]
for k, v in r["calibration"].items():
    L.append(f"| {k} | {v['temperature']:.3f} | {v['ece_heldout_before']:.4f} | {v['ece_heldout_after']:.4f} |")
L += ["", "T=0.050 is the lower edge of the search grid (the model is under-confident); see FINDINGS.", "",
      "## Unknown/novelty detection: signal ablation", "", "Rates are fractions of each set. `fabrication` = out-of-scope/OOD task answered with status KNOWN (lower is better).", ""]
sets = list(next(iter(r["detection_ablation"].values())))
L += ["| set | metric | " + " | ".join(r["detection_ablation"]) + " |", "|---|---|" + "---|" * len(r["detection_ablation"])]
for s in sets:
    first = r["detection_ablation"]["keyword"][s]
    if "fabrication_rate" in first:
        for m in ("fabrication_rate", "answered_uncertain", "needs_help"):
            L.append(f"| {s} | {m} | " + " | ".join(f"{d[s][m]:.3f}" for d in r["detection_ablation"].values()) + " |")
    else:
        for m in ("answered_known", "accuracy_known_only", "accuracy_all_answers", "needs_help"):
            L.append(f"| {s} | {m} | " + " | ".join(f"{d[s][m]:.3f}" for d in r["detection_ablation"].values()) + " |")
e = r["escalation"]
L += ["", "## Escalation over a task stream", "",
      f"Teacher calls: {e['teacher_calls_total']} over {e['tasks']} tasks. Related re-encounters answered locally without the teacher: {e['related_encounters_local']}/{e['related_encounters_total']}.",
      f"Offline: unknown task -> {e['offline_unknown']['result']} (teacher called while offline: {e['offline_unknown']['teacher_called_while_offline']}); known capability still answers offline: {e['offline_known_still_answers']}.", "",
      "| kind | intent | result | path | teacher called |", "|---|---|---|---|---|"]
for s in e["stream"]:
    L.append(f"| {s['kind']} | {s['intent']} | {s['result']} | {' > '.join(s['path'])} | {s['teacher_called']} |")
L += ["", "## Hostile / malformed teacher output", "", "| case | result | capabilities installed afterwards |", "|---|---|---|"]
for k, v in r["hostile_teacher_outputs"].items():
    L.append(f"| {k} | {v['result']} | {v['capabilities_after']} |")
p = r["privacy"]
L += ["", "## Privacy filter", "", f"Request actually sent: `{json.dumps(p['sent_request'])}`", f"Leaked secrets: {p['leaks']}", "",
      "## Information classification (rule baseline)", "", f"Tuned set: {r['information_classification']['accuracy']:.2f} on {r['information_classification']['n']} items (written together with the rules: not evidence).",
      f"Harder set: {r['information_classification']['tricky_set']['accuracy']:.2f} on {r['information_classification']['tricky_set']['n']} items; errors: {r['information_classification']['tricky_set']['errors']}", ""]
(ROOT / "benchmarks/reports/phase5.md").write_text("\n".join(L)); print("\n".join(L[:40]))
