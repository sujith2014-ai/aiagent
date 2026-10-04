"""Phase 6: real-provider path (mock HTTP), teacher-rule fault matrix, selective learning (router_update, adapt_existing,
new_module) and automated candidate evaluation."""
from __future__ import annotations
import json, shutil, sys, time, platform, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import Cli, ROOT
from packages.capbuild import keygen, write_trust
from integrations.teacher.simulator import TeacherSimulator, splits
from integrations.teacher.provider import HelpRequest
from integrations.teacher.http_providers import OpenAICompatProvider, AnthropicProvider
from integrations.teacher import mock_server
from integrations.escalation import Escalator
from training.service import BuildService
from training.strategies import SelectiveLearner
from training.learning_package.spec import TaskSpec, sample_spec
from scripts.run_phase5 import CANON, PARA, DIM

RUN = ROOT / "runs" / "phase6"
INTENTS = {"compare_numbers": ("compare these two numbers", [0.1, 0.9]), "point_region": ("is this point inside the circular region", [0.1, 0.1]),
           "argmax_position": ("find the position of the largest value", [0.1, 0.2, 0.9, 0.3])}


def env_examples(task, n=200):
    ex = splits(task)["eval"]
    return ex[0][:n], ex[1][:n]


def fresh(name, trust, teacher=None, learn=True):
    teacher = teacher or TeacherSimulator()
    svc = BuildService(RUN / "keys", "build-svc-1", RUN / f"build_{name}", teacher)
    cli = Cli(RUN / f"rt_{name}", trust)
    esc = Escalator(cli, teacher, svc, RUN / f"esc_{name}")
    if learn:
        for t, (intent, x) in INTENTS.items():
            r = esc.solve(intent, x); assert r["result"] == "ANSWER", r
    return teacher, svc, cli, esc


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keygen("build-svc-1", RUN / "keys"); trust = RUN / "trust.json"
    write_trust(trust, {"build-svc-1": (RUN / "keys/build-svc-1.public").read_text()})
    report = {"benchmark_format": "bench/1", "experiment": "phase6-selective-learning", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d")}

    # ---- A. teacher-rule fault matrix (spec mode, environment-verified) ----
    matrix = {}
    for fault in [None, "contradicts_examples", "malicious_expr", "plausible_wrong", "subtle_wrong"]:
        matrix[str(fault)] = {}
        for task, (intent, x) in INTENTS.items():
            T = TeacherSimulator(mode="spec", fault=fault)
            _, _, cli, esc = fresh(f"f_{fault}_{task}", trust, T, learn=False)
            r = esc.solve(intent, x, examples=env_examples(task))
            matrix[str(fault)][task] = {"result": r["result"], "reason": r.get("reason", "")[:110]}
    report["teacher_fault_matrix"] = matrix
    print("fault matrix:", {f: {t: v["result"] for t, v in d.items()} for f, d in matrix.items()})

    # ---- B. real provider code over HTTP (mock endpoints) ----
    http = {}
    backend = TeacherSimulator(mode="spec")
    srv, url, seen = mock_server.start(backend)
    prov = OpenAICompatProvider(url, "mock-qwen", api_key_env="MOCK_KEY")
    import os; os.environ["MOCK_KEY"] = "sk-test-not-a-real-key"
    svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build_http", backend); cli = Cli(RUN / "rt_http", trust)
    esc = Escalator(cli, prov, svc, RUN / "esc_http")
    r = esc.solve(*INTENTS["compare_numbers"], examples=env_examples("compare_numbers"))
    http["openai_compat_end_to_end"] = {"result": r["result"], "path": r["path"], "requests": len(seen),
                                        "auth_header_sent": "authorization" in seen[0]["headers"], "request_body_fields": sorted(json.loads(seen[0]["body"]["messages"][-1]["content"]))}
    srv.shutdown()
    srv, url, seen = mock_server.start(backend)
    ap = AnthropicProvider("mock-claude", base_url=url, api_key_env="MOCK_KEY")
    cli2 = Cli(RUN / "rt_http2", trust); esc2 = Escalator(cli2, ap, BuildService(RUN / "keys", "build-svc-1", RUN / "build_http2", backend), RUN / "esc_http2")
    r = esc2.solve(*INTENTS["point_region"], examples=env_examples("point_region"))
    http["anthropic_style_end_to_end"] = {"result": r["result"], "path": r["path"], "api_key_header": "x-api-key" in seen[0]["headers"]}
    srv.shutdown()
    srv, url, seen = mock_server.start(backend, fail=True)
    cli3 = Cli(RUN / "rt_http3", trust); esc3 = Escalator(cli3, OpenAICompatProvider(url, "m"), BuildService(RUN / "keys", "build-svc-1", RUN / "build_http3", backend), RUN / "esc_http3")
    r = esc3.solve(*INTENTS["compare_numbers"])
    http["server_error_503"] = {"result": r["result"], "teacher_calls_counted": esc3.teacher_calls, "queued": (RUN / "esc_http3/pending_help.jsonl").exists()}
    srv.shutdown()
    srv, url, seen = mock_server.start(backend, garbage="Sure! Here's a capability: DROP TABLE users;")
    cli4 = Cli(RUN / "rt_http4", trust); esc4 = Escalator(cli4, OpenAICompatProvider(url, "m"), BuildService(RUN / "keys", "build-svc-1", RUN / "build_http4", backend), RUN / "esc_http4")
    r = esc4.solve(*INTENTS["compare_numbers"])
    http["garbage_text_response"] = {"result": r["result"], "capabilities_installed": len(cli4.list())}
    srv.shutdown()
    http["live_endpoint_validation"] = "PENDING: requires network egress and an API key (see docs/RUNBOOKS.md)"
    report["http_providers"] = http
    print("http:", {k: (v["result"] if isinstance(v, dict) and "result" in v else v) for k, v in http.items()})

    # ---- C. router_update vs per-task rerouting on paraphrases ----
    def paraphrase_run(persist):
        T, svc_, cli_, esc_ = fresh(f"para_{persist}", trust)
        esc_.persist_router_updates = persist
        before = esc_.teacher_calls
        rows = []
        for rep in range(3):
            for t in DIM:
                ex = splits(t)["eval"]
                for k, ph in enumerate(PARA[t]):
                    c0 = esc_.teacher_calls
                    r = esc_.solve(ph, ex[0][5 + rep * 4 + k], examples=env_examples(t))
                    rows.append({"round": rep, "intent": ph, "result": r["result"], "teacher_called": esc_.teacher_calls > c0, "path": r["path"],
                                 "router_update": r.get("router_update")})
        return {"teacher_calls_after_learning": esc_.teacher_calls - before, "tasks": len(rows), "router_updates_accepted": esc_.router_updates,
                "per_round_teacher_calls": [sum(1 for x in rows if x["round"] == rd and x["teacher_called"]) for rd in range(3)],
                "answered": sum(1 for x in rows if x["result"] == "ANSWER"), "rows": rows}
    report["paraphrase_teacher_dependency"] = {"without_router_update": paraphrase_run(False), "with_router_update": paraphrase_run(True)}
    for k, v in report["paraphrase_teacher_dependency"].items():
        print(f"paraphrases {k}: teacher calls {v['teacher_calls_after_learning']}/{v['tasks']} per round {v['per_round_teacher_calls']} answered {v['answered']}")

    # ---- D. domain extension: compare_numbers must also work on a 4x wider numeric range ----
    T, svc_, cli_, esc_ = fresh("ext", trust)
    wide = {"capability_id": "compare_numbers", "description": "Compare two numbers (wide range)", "keywords": ["compare", "numbers"], "labels": ["LESS", "EQUAL", "GREATER"],
            "input_dim": 2, "domain": [{"lo": 0.0, "hi": 79 / 19, "grid": 80}] * 2, "label_expr": "0 if x0 < x1 else 1 if x0 == x1 else 2",
            "worked_examples": [{"x": [0.5, 3.0], "y": 0}, {"x": [3.0, 3.0], "y": 1}, {"x": [4.0, 1.0], "y": 2}], "n_train": 1500, "n_val": 300}
    spec = TaskSpec.from_dict(wide)
    venv = sample_spec(spec, 300, 4242, unique=True)
    old_eval = splits("compare_numbers")["eval"]
    before = cli_.solve("compare these two numbers", [3.1, 2.6])
    learner = SelectiveLearner(svc_, cli_)
    # ablation (dry runs, nothing installed): naive fine-tune without replay
    naive = svc_.adapt("compare_numbers", __import__("training.learning_package.spec", fromlist=["x"]).spec_to_learning_package(spec, "t"), venv, old_eval, "9.9.9", replay=False, build=False)
    log = learner.extend_domain("compare_numbers", spec, venv, old_eval)
    after = cli_.solve("compare these two numbers", [3.1, 2.6])
    old_after = cli_.regression("compare_numbers")
    report["domain_extension_compare_wide"] = {"before_extension_outcome": before.get("reason_code") or before.get("result"), "decision": log,
                                               "naive_finetune_no_replay_ablation": {**naive.report, "would_pass": naive.promoted, "reason": naive.reason},
                                               "after_extension": {"result": after["result"], "label": after.get("label"), "expected": "GREATER"},
                                               "bundled_regression_after": old_after}
    print("extension chosen:", log["chosen"], [(a["strategy"], a.get("passed")) for a in log["attempts"]], "naive:", naive.reason)

    # ---- E. conflicting rule change: point_region now means a *smaller* circle (r^2 < 0.2) ----
    T, svc_, cli_, esc_ = fresh("conflict", trust)
    small = {"capability_id": "point_region", "description": "Inside the small inner circular region", "keywords": ["small", "inner", "core", "circle", "point", "inside", "region"],
             "labels": ["INSIDE", "OUTSIDE"], "input_dim": 2, "domain": [{"lo": -1.0, "hi": 1.0}] * 2, "label_expr": "0 if x0*x0 + x1*x1 < 0.2 else 1",
             "worked_examples": [{"x": [0.0, 0.0], "y": 0}, {"x": [0.9, 0.9], "y": 1}, {"x": [0.1, 0.2], "y": 0}], "n_train": 1500, "n_val": 300}
    spec = TaskSpec.from_dict(small)
    venv = sample_spec(spec, 300, 4343, unique=True)
    old_eval = splits("point_region")["eval"]
    learner = SelectiveLearner(svc_, cli_)
    log = learner.extend_domain("point_region", spec, venv, old_eval, new_cap_keywords=small["keywords"])
    new_id = log.get("new_capability_id")
    routed = {}
    if new_id:
        routed["old_intent"] = cli_.solve("is this point inside the circular region", [0.5, 0.0])
        routed["new_intent"] = cli_.solve("is this point inside the small inner region", [0.5, 0.0])
    report["conflicting_rule_change"] = {"decision": log, "routing_after": {k: {"capability": v.get("capability_id"), "status": v.get("status"), "label": v.get("label")} for k, v in routed.items()},
                                         "probe": "(0.5, 0.0) has r^2=0.25: INSIDE under the old rule (r^2<0.5), OUTSIDE under the new rule (r^2<0.2)"}
    print("conflict chosen:", log["chosen"], [(a["strategy"], a.get("passed"), a.get("reason") or "") for a in log["attempts"]], "routing:", {k: v["capability"] for k, v in report["conflicting_rule_change"]["routing_after"].items()})
    out = ROOT / "benchmarks/reports/phase6.json"; out.write_text(json.dumps(report, indent=1, default=str)); print("wrote", out)


if __name__ == "__main__":
    main()
