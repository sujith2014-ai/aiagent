"""Live OpenClaw validation (NOT run in the research container: egress proxy blocks search providers, no credentials).
Usage:
  OPENCLAW_BIN=/path/to/openclaw.mjs python3 scripts/live_openclaw_check.py [--search-provider duckduckgo] [--query "..."] [--fetch-url https://docs.example.org/x]
What it does: lists providers, runs ONE search through the policy broker and prints the evidence (with hash/provenance), optionally fetches one
allowlisted URL. It never starts an OpenClaw agent turn. Record the output in docs/FINDINGS.md (see docs/RUNBOOKS.md R3)."""
import argparse, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import AICLI, ROOT
from integrations.openclaw.broker import ActionBroker
from integrations.openclaw.openclaw_cli import OpenClawCli, OpenClawActionProvider
from integrations.openclaw.action import ActionDenied, ActionUnavailable

ap = argparse.ArgumentParser()
ap.add_argument("--search-provider", default=None); ap.add_argument("--query", default="how to compare two numbers"); ap.add_argument("--fetch-url", default=None)
ap.add_argument("--policy", default=str(ROOT / "integrations/openclaw/policy.default.json"))
a = ap.parse_args()
cmd = ["node", os.environ["OPENCLAW_BIN"]] if os.environ.get("OPENCLAW_BIN", "").endswith(".mjs") else [os.environ.get("OPENCLAW_BIN", "openclaw")]
cli = OpenClawCli(command=cmd, profile="aiagent-live", search_provider=a.search_provider, timeout=120)
print("providers:", json.dumps(cli.providers())[:600])
d = Path(tempfile.mkdtemp()); broker = ActionBroker(AICLI, Path(a.policy), OpenClawActionProvider(cli), d / "audit.jsonl")
for label, fn in (("search", lambda: broker.search("live1", a.query)), ("fetch", (lambda: broker.fetch("live1", a.fetch_url)) if a.fetch_url else None)):
    if fn is None:
        continue
    try:
        print(label, "->", json.dumps([e.public() for e in fn()], indent=1)[:2500])
    except (ActionDenied, ActionUnavailable) as e:
        print(label, "-> NOT PERFORMED:", e)
print("audit:", (d / "audit.jsonl").read_text())
