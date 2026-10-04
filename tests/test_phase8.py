"""Phase 8 gates: policy boundary, broker audit, OpenClaw adapter contract, research loop with provenance, containment."""
import ast, hashlib, json, os, re, stat, subprocess, sys, textwrap
from pathlib import Path
import pytest
from conftest import ROOT
from scripts.cli import Cli, AICLI
from integrations.escalation import Escalator
from integrations.openclaw.action import ActionDenied, NeedsApproval, ActionUnavailable
from integrations.openclaw.broker import ActionBroker
from integrations.openclaw.fixture import FixtureProvider
from integrations.openclaw.openclaw_cli import OpenClawCli, OpenClawActionProvider, classify_failure
from integrations.teacher.provider import TeacherProvider
from integrations.teacher.simulator import TeacherSimulator
from packages.capbuild import keygen, write_trust
from training.service import BuildService

from scripts.phase8_fixtures import GOOD_PAGE, GRADE, INJECTION, grade_examples, Obedient

POLICY = ROOT / "integrations/openclaw/policy.default.json"


@pytest.fixture
def lab(tmp_path):
    keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
    pol = tmp_path / "policy.json"; pol.write_text(POLICY.read_text())
    def make(name, provider=None, teacher=None, online=True):
        T = teacher or TeacherSimulator()
        svc = BuildService(tmp_path / "keys", "k", tmp_path / f"b_{name}", T)
        cli = Cli(tmp_path / f"rt_{name}", tmp_path / "trust.json")
        broker = ActionBroker(AICLI, pol, provider, tmp_path / f"audit_{name}.jsonl", tmp_path / "approvals.json")
        esc = Escalator(cli, T, svc, tmp_path / f"e_{name}", broker=broker, online=online)
        return T, svc, cli, broker, esc
    make.policy, make.dir = pol, tmp_path
    return make


# ---------------------------------------------------------------------------------------------- policy boundary
@pytest.mark.parametrize("action,params,effect", [
    ("web.search", {"query": "grade band thresholds"}, "allow"),
    ("web.search", {"query": "x" * 300}, "deny"),
    ("web.search", {"query": "my password=hunter2"}, "deny"),
    ("web.fetch", {"url": "https://docs.example.org/page"}, "allow"),
    ("web.fetch", {"url": "https://en.wikipedia.org/wiki/Grade"}, "allow"),
    ("web.fetch", {"url": "https://evil.example.com/?q=1"}, "deny"),
    ("web.fetch", {"url": "http://docs.example.org/page"}, "deny"),
    ("web.fetch", {"url": "https://user:pw@docs.example.org/"}, "deny"),
    ("web.fetch", {"url": "https://169.254.169.254/latest/meta-data"}, "deny"),
    ("web.fetch", {"url": "https://localhost:8080/admin"}, "deny"),
    ("shell.exec", {"cmd": "ls"}, "deny"),
    ("tool.anything", {}, "deny"),
    ("browser.click", {"selector": "#x"}, "deny"),
    ("fs.read", {"path": "/home/u/notes.txt"}, "require_approval"),
])
def test_policy_decisions(lab, action, params, effect):
    _, _, _, broker, _ = lab("p")
    assert broker.check("t1", action, params)["effect"] == effect


def test_per_task_limit_and_fail_closed_on_engine_error(lab, tmp_path):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/g", GOOD_PAGE)]})
    _, _, _, broker, _ = lab("lim", provider=prov)
    broker.search("t1", "grade band thresholds")
    with pytest.raises(ActionDenied):
        broker.search("t1", "grade band thresholds again")          # max_per_task = 1
    broker.search("t2", "grade band thresholds")                    # a different task has its own budget
    broken = ActionBroker(AICLI, tmp_path / "missing_policy.json", prov, tmp_path / "a.jsonl")
    assert broken.check("t", "web.search", {"query": "x"})["effect"] == "deny"        # fail closed


def test_approval_flow_is_operator_only_and_bound_to_the_exact_request(lab, tmp_path):
    _, _, _, broker, _ = lab("ap", provider=FixtureProvider())
    params = {"path": "/home/u/a.txt"}
    assert broker.check("t", "fs.read", params)["effect"] == "require_approval"
    req = json.dumps({"action": "fs.read", "params": params})
    out = subprocess.run([str(AICLI), "--root", "/x", "--trust", "/x", "approve", "--approvals", str(tmp_path / "approvals.json"), "--request", req, "--approver", "alice", "--ttl", "60"], capture_output=True, text=True)
    assert out.returncode == 0
    assert broker.check("t", "fs.read", params)["effect"] == "allow"
    assert broker.check("t", "fs.read", {"path": "/home/u/b.txt"})["effect"] == "require_approval"


def test_ai_facing_code_has_no_path_to_change_policy_or_approvals():
    banned_strings = {"approve"}
    for f in (ROOT / "integrations").rglob("*.py"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                assert n.value not in banned_strings, f
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open":
                if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and "w" in str(n.args[1].value):
                    target = ast.unparse(n.args[0])
                    assert "policy" not in target and "approvals" not in target, (f, target)


def test_audit_log_records_every_decision_and_never_raw_secrets(lab):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/g", GOOD_PAGE)]})
    _, _, _, broker, _ = lab("aud", provider=prov)
    broker.search("t1", "grade bands for bob@example.com")                     # personal data is redacted before it leaves
    with pytest.raises(ActionDenied):
        broker.search("t2", "grade bands with sk-abcdef1234567890 and password=hunter2")   # a query carrying secrets is refused, not just masked
    with pytest.raises(ActionDenied):
        broker.fetch("t1", "https://evil.example.com/steal")
    lines = [json.loads(l) for l in broker.audit_path.read_text().splitlines()]
    assert [l["result"] for l in lines] == ["ok", "denied", "denied"]
    text = broker.audit_path.read_text()
    for leak in ("bob@example.com", "sk-abcdef1234567890", "hunter2"):
        assert leak not in text
    assert prov.calls == [("search", "grade bands for [EMAIL]")]                # what the provider actually received
    assert lines[2]["rule"] == "fetch-allowlist" and lines[0]["provider"] == "fixture-simulated"


# ---------------------------------------------------------------------------------------------- OpenClaw adapter contract
def fake_openclaw(tmp_path, behaviour: str) -> list[str]:
    script = tmp_path / f"fake_{behaviour}.py"
    script.write_text(textwrap.dedent(f'''
        import sys, json, os, time
        behaviour = {behaviour!r}
        if behaviour == "env":
            print(json.dumps({{"ok": True, "capability": "web.search", "provider": "duckduckgo", "outputs": [{{"result": {{"results": [{{"title": "t", "url": "https://x.example/", "snippet": json.dumps({{"HOME": os.environ.get("HOME"), "keys": sorted(os.environ)}})}}]}}}}]}})); sys.exit(0)
        if behaviour == "success":
            print(json.dumps({{"ok": True, "capability": "web.search", "transport": "local", "provider": "duckduckgo", "attempts": [],
                "outputs": [{{"result": {{"query": "q", "results": [{{"title": "Grading", "url": "https://docs.example.org/g", "snippet": "{GOOD_PAGE}"}}]}}}}]}})); sys.exit(0)
        if behaviour == "fetch":
            print(json.dumps({{"ok": True, "capability": "web.fetch", "provider": "p", "outputs": [{{"result": {{"content": "{GOOD_PAGE}"}}}}]}})); sys.exit(0)
        if behaviour == "no_provider":
            sys.stderr.write("Error: web_search is disabled or no provider is available.\\n"); sys.exit(1)
        if behaviour == "network":
            sys.stderr.write("TypeError: fetch failed | Request was cancelled. | 0 | Proxy response (403) !== 200 when HTTP Tunneling | UND_ERR_ABORTED\\n"); sys.exit(1)
        if behaviour == "timeout":
            time.sleep(30)
        if behaviour == "garbage":
            print("<html>not json</html>"); sys.exit(0)
        if behaviour == "huge":
            print(json.dumps({{"ok": True, "outputs": [{{"result": "x" * 500000}}]}})); sys.exit(0)
        if behaviour == "noresults":
            print(json.dumps({{"ok": True, "provider": "p", "outputs": [{{"result": {{"results": []}}}}]}})); sys.exit(0)
    '''))
    return [sys.executable, str(script)]


@pytest.mark.parametrize("behaviour,kind", [("no_provider", "no_provider"), ("network", "network"), ("timeout", "timeout"), ("garbage", "bad_output"), ("huge", "too_large"), ("noresults", "no_results")])
def test_adapter_maps_failures_to_unavailable_never_to_evidence(tmp_path, behaviour, kind):
    cli = OpenClawCli(command=fake_openclaw(tmp_path, behaviour), timeout=2.0)
    with pytest.raises(ActionUnavailable) as e:
        OpenClawActionProvider(cli).search("grade")
    assert e.value.kind == kind


def test_adapter_parses_success_envelope_and_fetch(tmp_path):
    ev = OpenClawActionProvider(OpenClawCli(command=fake_openclaw(tmp_path, "success"))).search("grade")
    assert ev[0].source.endswith("https://docs.example.org/g") and "below 0.3" in ev[0].excerpt and ev[0].provider == "openclaw:duckduckgo" and not ev[0].simulated
    pg = OpenClawActionProvider(OpenClawCli(command=fake_openclaw(tmp_path, "fetch"))).fetch("https://docs.example.org/g")
    assert "below 0.3" in pg[0].excerpt


def test_adapter_does_not_inherit_the_callers_secrets_and_uses_an_isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret"); monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    state = tmp_path / "state"
    ev = OpenClawActionProvider(OpenClawCli(command=fake_openclaw(tmp_path, "env"), state_dir=state)).search("x")
    seen = json.loads(ev[0].excerpt.split(" ", 1)[1])
    assert "OPENAI_API_KEY" not in seen["keys"] and "GITHUB_TOKEN" not in seen["keys"]
    assert seen["HOME"] == str(state)                                    # HOME points at the isolated state dir


def test_failure_classifier_on_messages_observed_from_the_real_binary():
    assert classify_failure("Error: web_search is disabled or no provider is available.") == "no_provider"
    assert classify_failure("Error: web.fetch is disabled or no provider is available.") == "no_provider"
    assert classify_failure("TypeError: fetch failed | Request was cancelled. | 0 | Proxy response (403) !== 200 when HTTP Tunneling | UND_ERR_ABORTED") == "network"


@pytest.mark.skipif(not os.environ.get("OPENCLAW_BIN"), reason="set OPENCLAW_BIN=/path/to/openclaw.mjs to run against a real OpenClaw install (docs/RUNBOOKS.md R3)")
def test_real_openclaw_binary_contract(tmp_path):
    cli = OpenClawCli(command=["node", os.environ["OPENCLAW_BIN"]], profile="aiagent-test", state_dir=tmp_path / "home", timeout=120)
    prov = cli.providers()
    assert {"search", "fetch"} <= set(prov) and any(p["id"] == "duckduckgo" for p in prov["search"])
    with pytest.raises(ActionUnavailable) as e:
        OpenClawActionProvider(cli).search("x")                           # no search provider selected in a fresh profile
    assert e.value.kind in ("no_provider", "network")
    with pytest.raises(ActionUnavailable) as e:
        OpenClawActionProvider(cli).fetch("https://example.com/")
    assert e.value.kind in ("no_provider", "network")


# ---------------------------------------------------------------------------------------------- research loop
def run_grade(lab, name, page, **kw):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/grading", page)]})
    T, svc, cli, broker, esc = lab(name, provider=prov, **kw)
    r = esc.solve(GRADE, [0.5], examples=grade_examples())
    return T, svc, cli, broker, esc, prov, r


def test_research_loop_learns_a_capability_whose_rule_exists_only_in_documents(lab):
    T, svc, cli, broker, esc, prov, r = run_grade(lab, "ok", GOOD_PAGE)
    assert r["result"] == "ANSWER" and r["label"] == "MID" and r["path"] == ["local", "teacher", "openclaw", "learn"]
    assert r["teacher_calls"] == 2 and esc.openclaw_calls == 1 and r["research"]["simulated"] is True
    r2 = esc.solve("which grade band is this score", [0.9], examples=grade_examples())
    assert r2["label"] == "HIGH" and r2["path"] == ["local"] and esc.teacher_calls == 2 and esc.openclaw_calls == 1       # no teacher, no OpenClaw next time


def test_provenance_of_external_evidence_is_stored_in_the_signed_package(lab):
    import zipfile
    T, svc, cli, broker, esc, prov, r = run_grade(lab, "prov", GOOD_PAGE)
    cap = next((cli.root / "store").glob("grade_band*.cap"))
    m = json.loads(zipfile.ZipFile(cap).read("manifest.json"))
    lp = m["provenance"]["learning_package"]
    assert lp["unverified_internet_content"] is True and lp["evidence"][0]["sha256"] == hashlib.sha256(GOOD_PAGE.encode()).hexdigest()
    assert lp["evidence"][0]["simulated"] is True and "grade" in lp["label_expr"] or "x0" in lp["label_expr"]
    assert m["provenance"]["verified_by_environment"] is True


def test_wrong_evidence_is_not_ground_truth_environment_verification_rejects_it(lab):
    wrong = "Grading policy. Scores below 0.5 are LOW. Scores from 0.5 up to 0.9 are MID. Scores of 0.9 and above are HIGH."
    T, svc, cli, broker, esc, prov, r = run_grade(lab, "wrong", wrong)
    assert r["result"] == "NEEDS_HELP" and "verification" in r["reason"] and cli.list() == []


def test_unverifiable_research_result_is_not_installed(lab):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/grading", GOOD_PAGE)]})
    T, svc, cli, broker, esc = lab("noex", provider=prov)
    r = esc.solve(GRADE, [0.5])                                       # no environment examples
    assert r["result"] == "NEEDS_HELP" and "verified" in r["reason"] and cli.list() == []


@pytest.mark.parametrize("fail", ["no_provider", "network", "timeout"])
def test_unavailable_research_is_queued_and_never_pretended(lab, fail):
    prov = FixtureProvider(fail=fail)
    T, svc, cli, broker, esc = lab(f"un_{fail}", provider=prov)
    r = esc.solve(GRADE, [0.5], examples=grade_examples())
    assert r["result"] == "QUEUED_EXTERNAL" and fail in r["reason"] and "nothing was researched" in r["reason"] and cli.list() == []
    assert "openclaw" not in r["path"] and esc.openclaw_calls == 0
    q = [json.loads(l) for l in (esc.workdir / "pending_help.jsonl").read_text().splitlines()]
    assert q and q[0]["why"].startswith("research unavailable")


def test_no_broker_means_research_is_reported_as_needed_not_done(lab, tmp_path):
    T = TeacherSimulator(); svc = BuildService(tmp_path / "keys", "k", tmp_path / "b", T); cli = Cli(tmp_path / "rt", tmp_path / "trust.json")
    esc = Escalator(cli, T, svc, tmp_path / "e")
    assert esc.solve(GRADE, [0.5], examples=grade_examples())["result"] == "NEEDS_EXTERNAL"


def test_policy_denial_stops_research(lab):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/grading", GOOD_PAGE)]})
    T, svc, cli, broker, esc = lab("deny", provider=prov)
    broker.policy_path.write_text(json.dumps({"version": 1, "default": "deny", "rules": []}))      # operator revokes search
    r = esc.solve(GRADE, [0.5], examples=grade_examples())
    assert r["result"] == "NEEDS_HELP" and "denied by policy" in r["reason"] and prov.calls == [] and esc.openclaw_calls == 0


def test_only_one_research_round_per_task(lab):
    class Loop(TeacherProvider):
        name = "loop"
        def advise(self, req): return json.dumps({"action": "external_research", "research_query": "again and again"})
    prov = FixtureProvider(search_results={"again": [("https://docs.example.org/x", "text")]})
    T, svc, cli, broker, esc = lab("loop", provider=prov, teacher=Loop())
    r = esc.solve(GRADE, [0.5], examples=grade_examples())
    assert r["result"] == "NEEDS_HELP" and "one per task" in r["reason"] and len(prov.calls) == 1 and r["teacher_calls"] == 2


# ---------------------------------------------------------------------------------------------- containment: assume the teacher obeys injected text
@pytest.mark.parametrize("second,expect", [
    ({"action": "request_tool", "tool": "shell.exec"}, "denied by policy"),
    ({"action": "new_capability_spec", "spec": {"capability_id": "pwn", "description": "d", "keywords": ["k"], "labels": ["A", "B"], "input_dim": 1, "domain": [{"lo": 0.0, "hi": 1.0}],
                                                 "label_expr": "__import__('os').system('rm -rf /')", "worked_examples": [{"x": [0.1], "y": 0}] * 3}}, "spec rejected"),
    ({"action": "reroute", "reroute_intent": "x", "capability_id": "ghost"}, "rejected"),
    ({"action": "use_memory"}, "rejected"),
])
def test_compromised_teacher_cannot_escape_policy_or_gates(lab, second, expect):
    prov = FixtureProvider(search_results={"grade": [("https://docs.example.org/grading", INJECTION)]})
    T, svc, cli, broker, esc = lab("inj", provider=prov, teacher=Obedient(second))
    before = hashlib.sha256(broker.policy_path.read_bytes()).hexdigest()
    r = esc.solve(GRADE, [0.5], examples=grade_examples())
    assert r["result"] == "NEEDS_HELP" and expect in r["reason"] and cli.list() == []
    assert hashlib.sha256(broker.policy_path.read_bytes()).hexdigest() == before          # policy untouched
    assert "ghost" not in str(cli.list())


def test_evidence_with_injected_text_is_passed_as_data_and_hashed(lab):
    T, svc, cli, broker, esc, prov, r = run_grade(lab, "inj_data", INJECTION)
    assert r["result"] == "ANSWER"                                            # the simulated teacher ignores instructions in evidence
    assert r["research"]["evidence"][0]["sha256"] == hashlib.sha256(INJECTION.encode()).hexdigest()


def test_exfiltration_via_fetch_is_blocked_by_domain_allowlist_and_secret_guard(lab):
    prov = FixtureProvider(pages={"https://docs.example.org/x?token=abc": "x"})
    _, _, _, broker, _ = lab("exf", provider=prov)
    for url in ["https://evil.example.com/collect?data=SECRET", "https://docs.example.org/x?token=abc", "https://docs.example.org.evil.com/", "https://127.0.0.1:9/"]:
        with pytest.raises(ActionDenied):
            broker.fetch("t", url)
    assert prov.calls == []
