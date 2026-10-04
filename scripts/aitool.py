#!/usr/bin/env python3
"""Guarded tool host CLI: the only entry point an OpenClaw skill (or anything else) should use to run installed tools.
  aitool.py --root REGISTRY_ROOT [--audit FILE] run TOOL_ID --input JSON
  aitool.py --root REGISTRY_ROOT list
Every run goes through the policy engine, signature + hash verification and the sandbox. There is deliberately no install/approve command here."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase13_lab import ToolLab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("tool_id"); r.add_argument("--input", required=True)
    sub.add_parser("list")
    a = ap.parse_args()
    lab = ToolLab(Path(a.root))
    if a.audit: lab.audit = Path(a.audit); lab.broker.audit_path = lab.audit
    if a.cmd == "list":
        print(json.dumps([{k: r[k] for k in ("tool_id", "version", "description", "keywords", "code_sha256")} for r in lab.registry.installed()], indent=1)); return 0
    try: inp = json.loads(a.input)
    except ValueError: print(json.dumps({"ok": False, "refused": "BAD_INPUT", "reason": "--input must be a JSON object"})); return 2
    if not isinstance(inp, dict): print(json.dumps({"ok": False, "refused": "BAD_INPUT", "reason": "--input must be a JSON object"})); return 2
    res = lab.host.call(a.tool_id, inp, task_id="openclaw-skill"); print(json.dumps(res)); return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
