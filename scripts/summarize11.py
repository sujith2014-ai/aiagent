"""Render benchmarks/reports/phase11.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase11.json").read_text())
L = [f"# Phase 11 benchmark summary ({r['date']}, {r['platform']})", "", f"Network: {r['network']}.", "", "## Server-side training and signed delivery (4 real tasks, constrained device profile)", "",
     "| task | learned | held-out acc | upload (JSON bytes) | package bytes | server train+sign s | device verify+install s | installed |", "|---|---|---|---|---|---|---|---|"]
for k, v in r["delivery"].items():
    L.append(f"| {k} | {v['learned']} | {v['heldout_accuracy']} | {v['upload_bytes_json']} | {v['package_bytes']} | {v['server_train_and_sign_s']} | {v['sync_verify_install_s']} | {v['installed_on_device']} |")
l = r["latency_us"]
L += ["", f"Signed catalog: {r['catalog_bytes']} bytes. Device holds a private key: **{r['device_holds_private_key']}**.", "", "## Local vs remote execution", "",
      f"Remote (loopback HTTP, server runs the core per request): p50 {l['remote_loopback_http']['p50']:.0f} us, p95 {l['remote_loopback_http']['p95']:.0f} us. Local, including the harness's process spawn per call: p50 {l['local_incl_process_spawn']['p50']:.0f} us. {l['note']}", "",
      "## Placement", ""] + [f"- **{k}**: {json.dumps(v)}" for k, v in r["placement"].items()]
o = r["offline"]
L += ["", "## Offline timeline", "", f"Known tasks answered while offline: {o['known_answered_offline']}/{o['known_total']}; unknown tasks queued (not pretended): {o['unknown_queued']}; teach while offline: {o['teach_while_offline']}; pending before reconnect: {o['pending_before_reconnect']}; flush while still offline: {o['flush_while_offline']}; flush after reconnect: {o['flush_after_reconnect']}.", "",
      "## Attacks on the delivery path", "", "| attack | defended | outcome | capabilities on device afterwards |", "|---|---|---|---|"]
for k, v in r["attacks"].items():
    L.append(f"| {k} | {v['defended']} | {json.dumps(v['outcome'])[:140]} | {v['device_capabilities_after']} |")
(ROOT / "benchmarks/reports/phase11.md").write_text("\n".join(L)); print("ok")
