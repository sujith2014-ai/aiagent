"""Render benchmarks/reports/phase8.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase8.json").read_text())
L = [f"# Phase 8 benchmark summary ({r['date']}, {r['platform']})", "", "## OpenClaw adapter vs the real binary (contract only; live research is PENDING)", ""]
real = r["openclaw_real_binary"]
for k, v in real.items():
    L.append(f"- **{k}**: {v}")
L += ["", "## Policy decisions (deterministic, default deny)", "", "| action | params | effect | rule | reason |", "|---|---|---|---|---|"]
for m in r["policy_matrix"]:
    L.append(f"| {m['action']} | `{m['params']}` | {m['effect']} | {m['rule']} | {m['reason']} |")
L += ["", "## Research loop (SIMULATED provider and teacher: shows plumbing, provenance and gates, not real research quality)", ""]
for k, v in r["research_scenarios"].items():
    L.append(f"- **{k}**: {json.dumps(v)}")
L += ["", "## Containment: a compromised teacher that obeys instructions injected into retrieved text", "", "| attack | result | capabilities installed | policy file unchanged |", "|---|---|---|---|"]
for k, v in r["containment"].items():
    if isinstance(v, dict) and "result" in v:
        L.append(f"| {k} | {v['result']} ({v['reason']}) | {v['installed']} | {v['policy_unchanged']} |")
L += ["", "Exfiltration-style URLs: " + json.dumps(r["containment"]["exfiltration_urls"]) + f"; provider calls actually made: {r['containment']['provider_calls_made']}", "",
      "## External dependence over a task stream", "", "| task kind | result | teacher calls | OpenClaw calls |", "|---|---|---|---|"]
for x in r["dependence_stream"]["rows"]:
    L.append(f"| {x['kind']} | {x['result']} | {x['teacher_calls']} | {x['openclaw_calls']} |")
d = r["dependence_stream"]
L += ["", f"Totals: {d['teacher_calls_total']} teacher calls and {d['openclaw_calls_total']} OpenClaw call over {d['tasks']} tasks; {d['tasks_resolved_locally']} resolved without any external help.", ""]
(ROOT / "benchmarks/reports/phase8.md").write_text("\n".join(L)); print("ok")
