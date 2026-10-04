"""Live teacher validation (NOT run in the research container: needs network egress + an API key).
Usage:
  OPENAI_API_KEY=... python3 scripts/live_teacher_check.py openai-compat https://api.example.com/v1 MODEL_NAME
  ANTHROPIC_API_KEY=... python3 scripts/live_teacher_check.py anthropic MODEL_NAME
  python3 scripts/live_teacher_check.py mock          # smoke test against the local mock endpoint
It asks the real model to teach `compare these two numbers`, verifies the returned rule against environment examples,
builds/installs the capability and prints the escalation record. The model only ever receives the minimal HelpRequest."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import Cli
from packages.capbuild import keygen, write_trust
from integrations.escalation import Escalator
from integrations.teacher.http_providers import OpenAICompatProvider, AnthropicProvider
from integrations.teacher.simulator import TeacherSimulator, splits
from training.service import BuildService


def main():
    kind = sys.argv[1]
    d = Path(tempfile.mkdtemp()); keygen("k", d / "keys"); write_trust(d / "trust.json", {"k": (d / "keys/k.public").read_text()})
    oracle = TeacherSimulator()            # only used for the build service's held-out API; labels for promotion come from the spec + environment examples
    srv = None
    if kind == "openai-compat":
        teacher = OpenAICompatProvider(sys.argv[2], sys.argv[3])
    elif kind == "anthropic":
        teacher = AnthropicProvider(sys.argv[2])
    else:
        from integrations.teacher import mock_server
        srv, url, _ = mock_server.start(TeacherSimulator(mode="spec")); teacher = OpenAICompatProvider(url, "mock", api_key_env=None)
    esc = Escalator(Cli(d / "rt", d / "trust.json"), teacher, BuildService(d / "keys", "k", d / "b", oracle), d / "esc")
    ex = splits("compare_numbers")["eval"]
    rec = esc.solve("compare these two numbers", [0.1, 0.9], examples=(ex[0], ex[1]))
    print(json.dumps(rec, indent=1))
    print("queue file:", (d / "esc/pending_help.jsonl").read_text() if (d / "esc/pending_help.jsonl").exists() else "none")
    if srv:
        srv.shutdown()


main()
