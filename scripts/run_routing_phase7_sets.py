"""Compare keyword / idf routers on the Phase 7 hand-written evaluation sets (3 capabilities; 27 in-scope paraphrases, 24 out-of-scope, 15 adversarial keyword-overlap) for continuity with F6/F16."""
import json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.run_phase7 as P7
from scripts.cli import ROOT, Cli
from packages.capbuild import keygen, write_trust
from integrations.teacher.simulator import TeacherSimulator
from integrations.escalation import Escalator
from training.service import BuildService

RUN = ROOT / "runs" / "routing_phase7_sets"
if RUN.exists(): shutil.rmtree(RUN)
RUN.mkdir(parents=True); P7.RUN = RUN
keygen("build-svc-1", RUN / "keys"); write_trust(RUN / "trust.json", {"build-svc-1": (RUN / "keys/build-svc-1.public").read_text()})
teacher = TeacherSimulator(); svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build", teacher)
cli = Cli(RUN / "rt", RUN / "trust.json"); esc = Escalator(cli, teacher, svc, RUN / "esc")
for t in ("compare_numbers", "point_region", "argmax_position"): P7.learn(t, esc)
rows = P7.router_eval_sets(random.Random(11)); out = {}
for name, router in (("keyword", None), ("idf:0.25", "idf:0.25"), ("idf:0.5", "idf:0.5")):
    out[name] = P7.eval_router(cli, rows, router, name.replace(":", "_"))
(ROOT / "benchmarks/reports/routing_phase7_sets.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:1500])
