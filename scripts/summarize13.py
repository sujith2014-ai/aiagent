"""Render benchmarks/reports/phase13.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase13.json").read_text())
g = r["growth"]
L = [f"# Phase 13 benchmark summary ({r['date']}, {r['platform']})", "", "**Caveats:** " + "; ".join(r["caveats"]) + ".", "",
     "## A. Ten missing actions through the full loop (gap -> generate -> gates -> operator approval -> install -> call)", "",
     f"Tools installed: {g['tools_installed']}/10; operator approvals issued: {g['operator_approvals_issued']}; own intent routed to the right tool: {g['own_intent_routed_correctly']}; off-topic intents wrongly routed: {g['off_topic_intents_wrongly_routed']}.", "",
     "| tool | gap before | attempts | first attempt rejected at | answer = reference | gap-to-answer s | call ms p50 | call ms p95 |", "|---|---|---|---|---|---|---|---|"]
for k, v in g["tasks"].items():
    L.append(f"| {k} | {v['gap_before']} | {v['attempts']} | {v['rejections'][0]['stage'] if v['rejections'] else '-'} | {v['answer_matches_reference']} | {v['growth_seconds']} | {v['call_ms_p50']} | {v['call_ms_p95']} |")
L += ["", "Call latency is dominated by starting the sandbox process (a fresh interpreter per call), not by the tool.", "",
      "## B. 23 adversarial or defective candidates under five sandbox configurations", "",
      "'harmful' = 15 candidates whose payload has a measurable effect (file written, file deleted, secret read, local listener contacted, environment read). Harm is judged from canaries, at evaluation time and again when the tool is called with its trigger input.", "",
      "| configuration | rejected before approval | harmful candidates that caused harm |", "|---|---|---|"]
for c, v in r["attacks"].items():
    L.append(f"| {c} | {v['rejected_before_approval']}/{v['total']} | {v['harmful_attacks_that_caused_harm_eval_or_call']}/{v['harmful_attacks_total']} |")
L += ["", "Stage that stopped each candidate (full gates):", "", "| candidate | stopped at |", "|---|---|"]
for k, v in r["attacks"]["full_gates"]["attacks"].items(): L.append(f"| {k} | {v['stopped_at']} |")
L += ["", "Candidates that still caused harm when layers are removed:", ""]
for c in list(r["attacks"])[1:]:
    harm = [k for k, v in r["attacks"][c]["attacks"].items() if v["harm_during_evaluation"] or v["harm_when_triggered_at_call_time"]]
    L.append(f"- **{c}**: {', '.join(harm) if harm else 'none'}")
L += ["", "## C. Approval and installation abuse", "", "| case | result | expected | ok |", "|---|---|---|---|"]
for k, v in r["abuse"].items():
    key = next(x for x in v if x not in ("expected", "as_expected")); L.append(f"| {k} | {v[key]} | {v['expected']} | {v['as_expected']} |")
d = r["static_false_positives"]
L += ["", "## D. Static analysis: cost in false positives", "", f"Benign snippets accepted: {d['accepted']}/{d['total']} ({', '.join(k for k, v in d['benign_snippets'].items() if not v)} rejected). Honest task sources accepted: {d['honest_task_sources_accepted']}/10; buggy ones accepted (static does not judge correctness): {d['buggy_task_sources_accepted']}/10.", "",
      "## E. OpenClaw", "", "```json", json.dumps(r["openclaw_skill"], indent=1), "```"]
(ROOT / "benchmarks/reports/phase13.md").write_text("\n".join(L)); print("ok")
