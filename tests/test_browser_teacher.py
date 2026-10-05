"""Browser-based teacher (no API key): OpenClaw's browser drives an AI chat website. Unit tests use a fake driver; end-to-end tests drive the REAL OpenClaw browser against a local mock chat site
(needs OPENCLAW_BIN and a Chromium; the site is a mock, so this validates the mechanism, not any real service)."""
import glob, json, os, secrets, socket, subprocess, time
from pathlib import Path
import pytest
from conftest import ROOT
from scripts.cli import AICLI, Cli
from integrations.teacher.browser_teacher import BrowserTeacher, SiteAdapter, TeacherNeedsHuman, extract_json, build_prompt
from integrations.teacher.http_providers import TeacherUnavailable
from integrations.teacher.provider import make_help_request
from integrations.teacher.response import parse_response, InvalidTeacherResponse
from integrations.teacher.simulator import TeacherSimulator, splits
from integrations.openclaw.broker import ActionBroker
from integrations.openclaw.browser import OpenClawBrowser, BrowserError, parse_snapshot


def req(intent="compare these two numbers"): return make_help_request(intent, 2, [], {}, "")


def test_extract_json_handles_prose_fences_and_nesting_and_rejects_non_json():
    assert json.loads(extract_json('Sure!\n```json\n{"action": "cannot_help", "rationale": "x"}\n```\nbye'))["action"] == "cannot_help"
    assert json.loads(extract_json('noise {not json} then {"a": {"b": [1, {"c": 2}]}} tail'))["a"]["b"][1]["c"] == 2
    with pytest.raises(TeacherUnavailable): extract_json("Roses are red, violets are blue")


def test_snapshot_parser_reads_roles_names_and_refs():
    refs = parse_snapshot('- generic [active] [ref=e1]:\n  - textbox "Message" [ref=e2]:\n    - /placeholder: Ask anything\n  - button "Send" [ref=e3]\n')
    assert [(r.role, r.name, r.ref) for r in refs] == [("generic", "", "e1"), ("textbox", "Message", "e2"), ("button", "Send", "e3")]


def test_the_prompt_is_one_line_and_carries_only_the_minimal_request():
    p = build_prompt(make_help_request("compare these two numbers", 2, ["a"], {"reason_code": "UNKNOWN"}, "x")); assert "\n" not in p and "REQUEST:" in p
    assert json.loads(p.split("REQUEST:", 1)[1])["task_intent"] == "compare these two numbers"


class FakeBrowser:
    def __init__(self, snap, replies=None): self.snap, self.replies, self.log, self.n = snap, replies or [], [], 0
    def open(self, url): self.log.append(("open", url)); return "t1"
    def snapshot(self): return self.snap
    def type(self, ref, text): self.log.append(("type", ref, text))
    def click(self, ref): self.log.append(("click", ref))
    def press(self, key): self.log.append(("press", key))
    def wait_ms(self, ms): pass
    def close(self, tab): self.log.append(("close", tab))
    def evaluate(self, fn):
        if "assistant" in fn: self.n += 1; return self.replies if self.n > 1 else []
        return False


CHAT_SNAP = '- generic [ref=e1]:\n  - textbox "Message" [ref=e2]\n  - button "Send" [ref=e3]\n'
LOGIN_SNAP = '- heading "Log in to continue" [ref=e1]\n- textbox "Email" [ref=e2]\n- textbox "Password" [ref=e3]\n- button "Log in" [ref=e4]\n'
CAPTCHA_SNAP = '- heading "Please verify you are human" [ref=e1]\n- button "I am not a robot" [ref=e2]\n'


def site(**kw): return SiteAdapter(name="fake", url="https://chat.example.org/", terms_acknowledged=True, poll_s=0.0, **kw)


def test_login_and_captcha_screens_stop_with_the_exact_manual_action_and_type_nothing():
    for snap in (LOGIN_SNAP, CAPTCHA_SNAP):
        fb = FakeBrowser(snap); t = BrowserTeacher(fb, site())
        with pytest.raises(TeacherNeedsHuman) as e: t.advise(req())
        assert "sign in yourself" in e.value.action and "https://chat.example.org/" in e.value.action and "do not give credentials" in e.value.action
        assert not [x for x in fb.log if x[0] in ("type", "press", "click")] and ("close", "t1") in fb.log


def test_nothing_is_sent_until_the_operator_acknowledged_the_site_terms():
    fb = FakeBrowser(CHAT_SNAP); t = BrowserTeacher(fb, SiteAdapter(name="x", url="https://chat.example.org/"))
    with pytest.raises(TeacherNeedsHuman, match="terms"): t.advise(req())
    assert fb.log == []


def test_a_normal_reply_is_returned_as_json_and_the_chat_tab_is_closed():
    fb = FakeBrowser(CHAT_SNAP, ['ok:\n```json\n{"action":"cannot_help","rationale":"no"}\n```'])
    out = BrowserTeacher(fb, site(stable_polls=1)).advise(req())
    assert parse_response(out, set()).action == "cannot_help" and [x[0] for x in fb.log][-1] == "close" and any(x[0] == "press" and x[1] == "Enter" for x in fb.log)


def test_no_reply_within_the_deadline_is_reported_as_unavailable_not_invented():
    fb = FakeBrowser(CHAT_SNAP, []); t = BrowserTeacher(fb, site(max_wait_s=0.05))
    with pytest.raises(TeacherUnavailable, match="no stable reply"): t.advise(req())


def test_the_policy_engine_gates_browser_access_by_url_and_the_operator_approves_once(tmp_path):
    pol = tmp_path / "pol.json"; pol.write_text(json.dumps({"version": 1, "default": "deny", "rules": [{"id": "b", "action": "browser.chat", "effect": "require_approval"}]}))
    ap = tmp_path / "ap.json"; broker = ActionBroker(AICLI, pol, None, tmp_path / "audit.jsonl", ap)
    fb = FakeBrowser(CHAT_SNAP, ['{"action":"cannot_help","rationale":"x"}']); t = BrowserTeacher(fb, site(stable_polls=1), broker)
    with pytest.raises(TeacherNeedsHuman, match="operator approval"): t.advise(req())
    assert fb.log == []
    r = {"action": "browser.chat", "params": {"url": "https://chat.example.org/"}}
    subprocess.run([str(AICLI), "--root", "/x", "--trust", "/x", "approve", "--approvals", str(ap), "--request", json.dumps(r), "--approver", "op", "--ttl", "60"], check=True, capture_output=True)
    assert parse_response(t.advise(req()), set()).action == "cannot_help"
    other = BrowserTeacher(FakeBrowser(CHAT_SNAP), SiteAdapter(name="o", url="https://other.example.org/", terms_acknowledged=True), broker)
    with pytest.raises(TeacherNeedsHuman): other.advise(req())                      # the approval is bound to the exact URL
    assert any(l["action"] == "browser.chat" for l in map(json.loads, (tmp_path / "audit.jsonl").read_text().splitlines()))


def test_default_policy_requires_approval_for_browser_chat(tmp_path):
    b = ActionBroker(AICLI, ROOT / "integrations/openclaw/policy.default.json", None, tmp_path / "a.jsonl")
    assert b.check("t", "browser.chat", {"url": "https://chat.example.org/"})["effect"] == "require_approval"
    assert b.check("t", "browser.chat", {"url": "http://127.0.0.1:1/"})["effect"] == "deny"               # loopback never, unless a rule says so


# ---------------------------------------------------------------------------------------------- real OpenClaw browser against a local mock chat site
OC = os.environ.get("OPENCLAW_BIN")
CHROME = os.environ.get("CHROME_BIN") or next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
needs_oc = pytest.mark.skipif(not (OC and CHROME), reason="set OPENCLAW_BIN and have a Chromium (CHROME_BIN or /opt/pw-browsers) to run the real-browser tests")


@pytest.fixture(scope="module")
def oc_browser(tmp_path_factory):
    home = tmp_path_factory.mktemp("ochome"); s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    tok = secrets.token_hex(16); env = {**os.environ, "HOME": str(home), "OPENCLAW_GATEWAY_TOKEN": tok}
    patch = {"gateway": {"mode": "local", "port": port, "auth": {"mode": "token", "token": tok}},
             "browser": {"enabled": True, "headless": True, "noSandbox": True, "executablePath": CHROME, "ssrfPolicy": {"dangerouslyAllowPrivateNetwork": True}}}      # test-only: the mock site is on loopback
    subprocess.run(["node", OC, "config", "patch", "--stdin"], input=json.dumps(patch), text=True, env=env, check=True, capture_output=True)
    gw = subprocess.Popen(["node", OC, "gateway", "--port", str(port)], env=env, stdout=open(home / "gw.log", "w"), stderr=subprocess.STDOUT)
    b = OpenClawBrowser(["node", OC], home=str(home), gateway_token=tok, timeout_s=60)
    for _ in range(60):
        try: b.start(); break
        except BrowserError: time.sleep(1)
    else: gw.kill(); pytest.fail("OpenClaw gateway/browser did not start: " + (home / "gw.log").read_text()[-500:])
    yield b
    gw.terminate()


def mock(advise, mode="normal", tool_advise=None):
    from scripts.mock_chat_site import start
    return start(advise, mode, tool_advise=tool_advise)


def local_site(url, **kw): return SiteAdapter(name="mock-chat", url=url, terms_acknowledged=True, busy_js="() => window.__busy === true", max_wait_s=40, **kw)


@needs_oc
def test_real_openclaw_browser_gets_a_teacher_reply_from_a_chat_page(oc_browser):
    sim = TeacherSimulator(mode="spec"); srv, url, st = mock(sim.advise)
    try:
        out = BrowserTeacher(oc_browser, local_site(url)).advise(req())
        resp = parse_response(out, set()); assert resp.action == "new_capability_spec" and resp.spec["capability_id"] == "compare_numbers"
        sent = st["prompts"][0]; assert "REQUEST:" in sent and set(json.loads(sent.split("REQUEST:", 1)[1])) == {"task_intent", "input_dim", "known_capabilities", "attempted_route", "failure", "available_tools", "evidence"}
    finally: srv.shutdown()


@needs_oc
def test_real_end_to_end_escalation_learns_a_capability_through_the_browser_teacher(oc_browser, tmp_path):
    from packages.capbuild import keygen, write_trust
    from integrations.escalation import Escalator
    from training.service import BuildService
    sim = TeacherSimulator(mode="spec"); srv, url, st = mock(sim.advise)
    try:
        keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
        esc = Escalator(Cli(tmp_path / "rt", tmp_path / "trust.json"), BrowserTeacher(oc_browser, local_site(url)), BuildService(tmp_path / "keys", "k", tmp_path / "b", TeacherSimulator()), tmp_path / "esc")
        ex = splits("compare_numbers")["eval"]
        r = esc.solve("compare these two numbers", [0.1, 0.9], examples=(ex[0], ex[1]))
        assert r["result"] == "ANSWER" and r["path"] == ["local", "teacher", "learn"] and r["label"] == "LESS" and len(st["prompts"]) == 1
        again = esc.solve("compare these two numbers", [0.9, 0.1]); assert again["path"] == ["local"] and again["label"] == "GREATER"      # learned: no second browser round trip
    finally: srv.shutdown()


@needs_oc
@pytest.mark.parametrize("mode,needle", [("login", "sign in yourself"), ("captcha", "sign in yourself")])
def test_real_login_wall_and_captcha_stop_for_a_human_and_the_request_is_queued(oc_browser, tmp_path, mode, needle):
    from packages.capbuild import keygen, write_trust
    from integrations.escalation import Escalator
    from training.service import BuildService
    srv, url, st = mock(TeacherSimulator(mode="spec").advise, mode)
    try:
        keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
        t = BrowserTeacher(oc_browser, local_site(url))
        with pytest.raises(TeacherNeedsHuman) as e: t.advise(req())
        assert needle in e.value.action and st["prompts"] == []
        esc = Escalator(Cli(tmp_path / "rt", tmp_path / "trust.json"), t, BuildService(tmp_path / "keys", "k", tmp_path / "b", TeacherSimulator()), tmp_path / "esc")
        r = esc.solve("compare these two numbers", [0.1, 0.9]); assert r["result"] == "QUEUED_OFFLINE" and "manual action required" in r["reason"]
        assert (tmp_path / "esc/pending_help.jsonl").exists() and esc.cli.list() == []
    finally: srv.shutdown()


@needs_oc
@pytest.mark.parametrize("mode", ["garbage", "injection"])
def test_real_hostile_or_useless_replies_install_nothing(oc_browser, tmp_path, mode):
    from packages.capbuild import keygen, write_trust
    from integrations.escalation import Escalator
    from training.service import BuildService
    srv, url, st = mock(TeacherSimulator(mode="spec").advise, mode)
    try:
        keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
        esc = Escalator(Cli(tmp_path / "rt", tmp_path / "trust.json"), BrowserTeacher(oc_browser, local_site(url)), BuildService(tmp_path / "keys", "k", tmp_path / "b", TeacherSimulator()), tmp_path / "esc")
        r = esc.solve("compare these two numbers", [0.1, 0.9]); assert r["result"] in ("NEEDS_HELP", "QUEUED_OFFLINE") and esc.cli.list() == []
        assert "rejected" in r.get("reason", "") or "no JSON" in r.get("reason", "")
    finally: srv.shutdown()


# ---------------------------------------------------------------------------------------------- tool generation through the browser (R5 without a model API key)
def _hidden_only(t):
    shown = [json.dumps(v) for v in t.visible]; return [c for c in t.spec_cases() if json.dumps(c["input"]) not in shown]       # spec cases the request did not already show as examples


def _tool_json(t, buggy):
    from integrations.toolgrowth.tasks import candidate
    c = candidate(t, buggy=buggy); return json.dumps({"description": c.description, "keywords": c.keywords, "source": c.source, "tests": c.tests})


def test_browser_tool_generator_sends_only_the_request_and_generic_feedback_and_survives_garbage():
    from integrations.toolgrowth.browser_generator import BrowserToolGenerator
    from integrations.toolgrowth.generator import ToolRequest
    from integrations.toolgrowth.tasks import BY_ID
    from integrations.teacher.browser_teacher import BrowserChat
    t = BY_ID["slugify"]
    fb = FakeBrowser(CHAT_SNAP, ['```json\n' + _tool_json(t, False) + '\n```'])
    g = BrowserToolGenerator(BrowserChat(fb, site(stable_polls=1)))
    c = g.generate(ToolRequest(t.task_id, t.intent, t.description, t.visible_cases()), 1, {"failed_stage": "spec", "message": "independent spec cases failed"})
    assert c.tool_id == "slugify" and c.version == "0.1.1" and c.generator.startswith("browser:")
    sent = g.seen_prompts[0]
    for case in _hidden_only(t): assert json.dumps(case["input"]) not in sent                       # hidden spec cases never leave
    assert "independent spec cases failed" in sent and "TOOL_REQUEST" in sent and "\n" not in sent
    bad = BrowserToolGenerator(BrowserChat(FakeBrowser(CHAT_SNAP, ['```json\n{"nope": 1}\n```']), site(stable_polls=1)))
    assert bad.generate(ToolRequest("x", "i", "d", []), 0, None) is None                           # malformed reply = a rejected attempt, not a crash


@needs_oc
def test_real_end_to_end_tool_growth_through_the_browser_generator_and_operator_approval(oc_browser, tmp_path):
    from scripts.phase13_lab import ToolLab
    from integrations.toolgrowth.browser_generator import BrowserToolGenerator
    from integrations.toolgrowth.generator import ToolRequest
    from integrations.toolgrowth.growth import GrowthLoop
    from integrations.toolgrowth.tasks import BY_ID
    from integrations.teacher.browser_teacher import BrowserChat
    t = BY_ID["roman_numerals"]
    srv, url, st = mock(TeacherSimulator().advise, tool_advise=lambda body: _tool_json(t, buggy=body["attempt"] == 0))
    try:
        L = ToolLab(tmp_path / "lab"); gen = BrowserToolGenerator(BrowserChat(oc_browser, local_site(url)))
        loop = GrowthLoop(L.pipeline, L.registry, L.host, L.index, gen)
        r = loop.handle(ToolRequest(t.task_id, t.intent, t.description, t.visible_cases()), t.spec_cases(), {"n": 1994}, operator=lambda c, rep: bool(L.operator_approve(c)))
        assert r["ok"], json.dumps(r["trace"])[:1500]
        assert r["output"] == {"result": "MCMXCIV"} and r["path"] == "grown"
        assert [a["status"] for a in r["trace"]["attempts"]] == ["REJECTED", "AWAITING_APPROVAL"] and r["trace"]["attempts"][0]["failed_stage"] == "tests"
        assert len(st["prompts"]) == 2 and "previous_attempt_failed" in st["prompts"][1] and "previous_attempt_failed" not in st["prompts"][0]
        for case in _hidden_only(t): assert json.dumps(case["input"]) not in " ".join(st["prompts"])
    finally: srv.shutdown()
