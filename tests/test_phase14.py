"""Phase 14 gates: the long-running benchmark's own machinery (world, teacher, stream, monitor, policies, baselines) and the router-alias fix it motivated."""
import json, collections
from pathlib import Path
import numpy as np, pytest
from conftest import ROOT
from scripts.phase14_lab import (World, BankTeacher, make_stream, DriftMonitor, UncertainAsHelp, CachingEscalator, ModularRun, Monolith, FAMILIES, DIM, NOUN_SYN, NOUNS, OOS)
from integrations.teacher.provider import make_help_request
from training.learning_package.spec import spec_to_learning_package


def test_world_has_36_capabilities_with_four_twins_and_valid_specs():
    w = World()
    assert len(w.caps) == 36 and len({c.cap_id for c in w.caps}) == 36 and len({c.noun for c in w.caps}) == 36
    twins = [c for c in w.caps if c.twin_of]; assert len(twins) == 4 and all(w.by_id[c.twin_of].params == c.params and w.by_id[c.twin_of].family == c.family for c in twins)
    assert set(NOUN_SYN) == set(NOUNS) and not set(NOUN_SYN.values()) & set(NOUNS)
    for c in w.caps:
        lp = spec_to_learning_package(w.spec(c.cap_id), "t"); assert lp.input_dim == DIM[c.family] and len(set(lp.train_y)) == len(lp.labels), c.cap_id


def test_every_family_drift_changes_a_meaningful_share_of_labels_and_is_deterministic():
    for fam in FAMILIES:
        w1, w2 = World(0), World(0); c = next(c for c in w1.caps if c.family == fam and not c.twin_of)
        xs, ys0 = w1.sample(c.cap_id, 400, 3); w1.drift(c.cap_id); w2.drift(c.cap_id)
        ys1 = [w1.label(c.cap_id, x) for x in xs]
        if fam != "peak":                                                                   # argmax has no parameter to drift
            assert np.mean(np.array(ys0) != np.array(ys1)) > 0.12, fam
        assert w1.by_id[c.cap_id].params == w2.by_id[c.cap_id].params


def test_stream_is_seeded_introduces_each_capability_first_and_schedules_drift():
    w = World(); a, b = make_stream(w, seed=5), make_stream(World(), seed=5)
    assert a == b and a != make_stream(World(), seed=6)
    seen = set(); intro = {}
    for e in a:
        if e["type"] == "new": assert e["cap"] not in intro; intro[e["cap"]] = e["t"]
        elif e["type"] in ("known", "paraphrase"): assert e["cap"] in intro, e
    assert len(intro) == 36 and [(e["t"], e["cap"]) for e in a if e["type"] == "drift"] == [(300, "boiler_alarm"), (420, "gear_zone"), (540, "mixer_zone")]
    assert collections.Counter(e["type"] for e in a)["oos"] > 20 and all(len(e["x"]) in (1, 2, 4) for e in a if e["type"] == "oos")


def test_teacher_identifies_by_noun_or_noun_synonym_and_follows_the_current_world():
    w = World(); t = BankTeacher(w)
    ask = lambda intent, known=(), failure="": json.loads(t.advise(make_help_request(intent, 2, list(known), {}, failure)))
    assert ask("check the pump zone")["action"] == "new_capability_spec" and ask("check the impeller region", known=["pump_zone"])["action"] == "reroute"
    assert ask("translate this paragraph into french")["action"] == "cannot_help" and ask("send an email to the team")["action"] == "request_tool"
    before = ask("check the boiler alarm", known=["boiler_alarm"], failure="drift")["spec"]["label_expr"]
    assert ask("check the boiler alarm", known=["boiler_alarm"])["action"] == "reroute"
    w.drift("boiler_alarm"); after = ask("check the boiler alarm", known=["boiler_alarm"], failure="drift: feedback accuracy dropped")["spec"]["label_expr"]
    assert before != after and t.calls == 6 and t.bytes_in > 0 and t.bytes_out > 0


def test_drift_monitor_needs_enough_evidence_and_resets():
    m = DriftMonitor(window=12, min_samples=8, threshold=0.7)
    assert not any(m.add("a", False) for _ in range(7)) and m.add("a", False)           # the 8th wrong outcome triggers
    m.reset("a"); assert not m.add("a", False)
    assert not any(m.add("b", i % 10 != 0) for i in range(30))                            # 90% correct never triggers


def test_uncertain_answers_are_escalated_but_known_ones_pass_and_attributes_delegate():
    class Cli:
        marker = 7
        def solve(self, i, x): return {"result": "ANSWER", "status": "UNCERTAIN" if i == "u" else "KNOWN", "capability_id": "c", "route_score": 0.3}
    p = UncertainAsHelp(Cli())
    assert p.solve("u", [0])["result"] == "NEEDS_HELP" and p.solve("u", [0])["reason_code"] == "UNCERTAIN_ROUTE" and p.solve("k", [0])["result"] == "ANSWER" and p.marker == 7


def mini(tmp_path, **kw):
    w = World(); r = ModularRun(tmp_path, w, **kw); return w, r


def ev(w, t, typ, cap, intent=None, fb=False):
    d = w.input_dim(cap); lo = -1.0 if w.by_id[cap].family in ("zone", "side") else 0.0
    return {"t": t, "type": typ, "cap": cap, "intent": intent or w.canonical(cap), "x": [round(lo + 0.37 * (1 + i) % 1.0 * (1.0 - lo), 4) for i in range(d)], "feedback": fb}


def test_negative_cache_stops_repeated_questions_for_unsupported_requests(tmp_path):
    counts = {}
    for cache in (False, True):
        w, r = mini(tmp_path / f"c{cache}", cache_refusals=cache, monitor=False)
        oos = {"t": 0, "type": "oos", "cap": None, "intent": "translate this paragraph into french", "x": [0.1, 0.2], "feedback": False}
        r.run([{**oos, "t": i} for i in range(5)], 100); counts[cache] = r.teacher.calls
    assert counts == {False: 5, True: 1}


def test_frame_words_are_not_aliased_when_the_filter_is_on(tmp_path):
    w, r = mini(tmp_path, monitor=False)
    stream = [ev(w, 0, "new", "boiler_alarm"), ev(w, 1, "new", "pump_zone"), ev(w, 2, "paraphrase", "boiler_alarm", "check the furnace warning"), ev(w, 3, "paraphrase", "pump_zone", "check the impeller region")]
    r.run(stream, 100)
    kw = {c["capability_id"]: set(c["keywords"]) for c in r.cli.list()}
    assert {"furnace", "warning"} <= kw["boiler_alarm"] and {"impeller", "region"} <= kw["pump_zone"] and "check" not in kw["boiler_alarm"] | kw["pump_zone"]
    assert r.cli.solve("check the furnace warning", [0.4])["capability_id"] == "boiler_alarm"


def test_drift_is_detected_from_feedback_and_repaired_with_a_new_version(tmp_path):
    w, r = mini(tmp_path)
    r.run([ev(w, 0, "new", "boiler_alarm")], 100)
    assert r.accuracy_all(1)["boiler_alarm"] > 0.9
    stream = [{"t": 1, "type": "drift", "cap": "boiler_alarm"}] + [{**ev(w, 2 + i, "known", "boiler_alarm", fb=True), "x": [0.34 + 0.02 * (i % 5)]} for i in range(30)]
    r.run(stream, 100)
    assert r.repairs and r.repairs[0]["result"] == "ANSWER" and r.repairs[0]["version_before"] != r.repairs[0]["version_after"] and 0 < r.repairs[0]["delay"] <= 20
    assert r.accuracy_all(99)["boiler_alarm"] > 0.9 and not r.repairs[0]["spurious"]


def test_monolith_baselines_have_fixed_size_and_respect_their_training_scope():
    w = World(); caps = [c.cap_id for c in w.caps]
    ft, jt = Monolith(caps, "finetune", epochs=5), Monolith(caps, "joint", epochs=5)
    n0 = ft.params(); xs, ys = w.sample("boiler_alarm", 100, 1)
    ft.learn("boiler_alarm", xs, ys); jt.learn("boiler_alarm", xs, ys); first_net = jt.net
    xs2, ys2 = w.sample("pump_zone", 100, 2); ft.learn("pump_zone", xs2, ys2); jt.learn("pump_zone", xs2, ys2)
    assert ft.params() == n0 == jt.params() and jt.net is not first_net and set(jt.data) == {"boiler_alarm", "pump_zone"} and 0 <= ft.accuracy("boiler_alarm", xs, ys) <= 1
    rp = Monolith(caps, "replay", epochs=3, buffer=10); rp.learn("boiler_alarm", xs, ys); rp.learn("pump_zone", xs2, ys2); assert len(rp.data["boiler_alarm"][0]) == 100       # buffer limits what is replayed, not what is stored


def test_window_stats_arithmetic():
    import scripts.run_phase14 as R
    rows = [{"t": 0, "type": "known", "cap": "a", "served": True, "correct": True, "teacher_calls": 0}, {"t": 1, "type": "known", "cap": "a", "served": True, "correct": False, "teacher_calls": 1},
            {"t": 2, "type": "oos", "cap": None, "served": False, "correct": None, "teacher_calls": 1}, {"t": 150, "type": "known", "cap": "a", "served": True, "correct": True, "teacher_calls": 0}]
    s = R.window_stats(rows, 0, 100)
    assert s == {"arrivals": 3, "teacher_calls": 2, "capability_arrivals": 2, "served": 2, "wrong": 1, "answered_without_teacher": 1}


REPORT = ROOT / "benchmarks/reports/phase14.json"


@pytest.mark.skipif(not REPORT.exists(), reason="run scripts/run_phase14.py first")
def test_report_is_internally_consistent():
    r = json.loads(REPORT.read_text())
    for name, a in r["arms"].items():
        assert sum(a["teacher_calls_by_action"].values()) == a["teacher_calls"], name
        assert a["served"] + a["unserved_capability_arrivals"] == a["capability_arrivals"], name
        assert sum(w["teacher_calls"] for w in a["windows_of_100"]) <= a["teacher_calls"], name        # repairs happen outside arrival windows
        assert all(c2["active_modules"] >= c1["active_modules"] - 0 for c1, c2 in zip(a["checkpoints"], a["checkpoints"][1:])), name
    assert r["teacher_always"]["teacher_calls"] > r["arms"]["base"]["teacher_calls"]
