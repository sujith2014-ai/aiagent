"""Phase 5: unknown/novelty detection (with signal ablation), calibration, teacher escalation loop with
privacy/offline handling, information classification, teacher-dependency over a task stream."""
from __future__ import annotations
import json, random, shutil, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import Cli, ROOT
from packages.capbuild import keygen, write_trust
from integrations.teacher.simulator import TeacherSimulator, splits, TASKS
from integrations.teacher.provider import HelpRequest, TeacherProvider, redact
from integrations.escalation import Escalator
from training.service import BuildService
from training.information import classify_information

RUN = ROOT / "runs" / "phase5"
CANON = {"compare_numbers": "compare these two numbers", "point_region": "is this point inside the circular region",
         "argmax_position": "find the position of the largest value"}
PARA = {"compare_numbers": ["comparison relation of numbers", "which of these numbers is bigger", "order the two values",
                            "is the first one greater than, smaller than or the same as the second"],
        "point_region": ["geometry check point inside circle", "does this coordinate fall within the disc",
                         "classify the point as inside or outside", "locate point relative to the circular region"],
        "argmax_position": ["which index holds the maximum", "where is the biggest entry", "pick the winning slot", "position of the highest value"]}
OOS = {2: ["translate this sentence to french", "add these two numbers", "is the first number prime", "what is the weather today",
           "summarise this article", "sort the list alphabetically", "convert celsius to fahrenheit", "who won the match"],
       4: ["average of four numbers", "compute the sum of the values", "what time is it", "classify the sentiment of this review",
           "reverse the list", "detect spam in this email", "estimate the age of the person", "translate to german"]}
ADV = {2: ["compare the sizes of two images", "compare prices of two shops", "is this point on the map inside the country",
           "region code lookup for this postcode", "compare two numbers written in roman numerals"],
       4: ["find the largest file in the folder", "highest maximum temperature index in the report", "position of the largest city",
           "which index has the maximum latency in the log"]}
DIM = {"compare_numbers": 2, "point_region": 2, "argmax_position": 4}


def rnd_input(dim, rng):
    return [round(rng.uniform(0, 1), 4) for _ in range(dim)]


def ood_input(task, rng):
    d = DIM[task]
    if task == "point_region":
        return [rng.uniform(1.5, 5) * rng.choice([-1, 1]) for _ in range(d)]
    return [rng.uniform(1.5, 6) * rng.choice([1, -0.5]) for _ in range(d)]


def write(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def build_sets(rng):
    sets = {"S1_in_scope_canonical": [], "S2_in_scope_paraphrase": [], "S3_in_scope_intent_ood_input": [],
            "S4_out_of_scope": [], "S5_adversarial_keyword_overlap": [], "S6_boundary_points": []}
    for t in DIM:
        ex, ey = splits(t)["eval"]
        for i in range(min(150, len(ex))):
            sets["S1_in_scope_canonical"].append({"intent": CANON[t], "input": ex[i], "expected_index": ey[i], "expected_capability": t})
        for i in range(min(120, len(ex))):
            sets["S2_in_scope_paraphrase"].append({"intent": PARA[t][i % 4], "input": ex[-1 - i], "expected_index": ey[-1 - i], "expected_capability": t})
        for i in range(60):
            sets["S3_in_scope_intent_ood_input"].append({"intent": CANON[t], "input": ood_input(t, rng)})
    for dim, intents in OOS.items():
        for i in range(60):
            sets["S4_out_of_scope"].append({"intent": intents[i % len(intents)], "input": rnd_input(dim, rng)})
    for dim, intents in ADV.items():
        for i in range(60):
            sets["S5_adversarial_keyword_overlap"].append({"intent": intents[i % len(intents)], "input": rnd_input(dim, rng)})
    for i in range(300):  # points near the circle boundary: genuinely hard in-scope cases
        a = rng.uniform(0, 6.2832); r = (0.5 ** 0.5) + rng.uniform(-0.06, 0.06)
        import math
        x, y = r * math.cos(a), r * math.sin(a)
        sets["S6_boundary_points"].append({"intent": CANON["point_region"], "input": [x, y], "expected_index": 0 if x * x + y * y < 0.5 else 1, "expected_capability": "point_region"})
    return sets


def metrics(name, rows, results):
    n = len(rows); ans = [(r, x) for r, x in zip(rows, results) if not x.get("needs_help")]
    known = [(r, x) for r, x in ans if x["status"] == "KNOWN"]
    unc = [(r, x) for r, x in ans if x["status"] == "UNCERTAIN"]
    out = {"n": n, "answered_known": len(known) / n, "answered_uncertain": len(unc) / n, "needs_help": (n - len(ans)) / n}
    if rows and "expected_index" in rows[0]:
        corr = lambda lst: sum(1 for r, x in lst if x["label_index"] == r["expected_index"] and x["capability"] == r["expected_capability"]) / max(1, len(lst))
        out["accuracy_all_answers"] = corr(ans)
        out["accuracy_known_only"] = corr(known)
        out["correct_known_fraction_of_all"] = sum(1 for r, x in known if x["label_index"] == r["expected_index"] and x["capability"] == r["expected_capability"]) / n
        out["error_rate_known_only"] = 1 - out["accuracy_known_only"] if known else None
        out["error_rate_all_answers"] = 1 - out["accuracy_all_answers"] if ans else None
    else:
        out["fabrication_rate"] = len(known) / n   # out-of-scope task answered with status KNOWN
        out["flagged_or_rejected_rate"] = 1 - len(known) / n
    return out


class Garbage(TeacherProvider):
    name = "garbage"
    def __init__(self, text): self.text = text
    def advise(self, req): return self.text


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keygen("build-svc-1", RUN / "keys"); trust = RUN / "trust.json"
    write_trust(trust, {"build-svc-1": (RUN / "keys/build-svc-1.public").read_text()})
    teacher = TeacherSimulator(); svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build", teacher)
    rt = Cli(RUN / "rt", trust); known = set(); report = {"benchmark_format": "bench/1", "experiment": "phase5-detection-and-escalation",
        "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"), "calibration": {}}
    for intent, dim in [("compare numbers relation", 2), ("point inside region", 2), ("largest value position", 4)]:
        lp = teacher.respond(HelpRequest(intent, dim)); b = svc.build(lp, "0.1.0", known); assert b.promoted
        assert rt.import_caps(str(b.cap_path))[0]["activated"]; known.add(lp.capability_id)
        report["calibration"][lp.capability_id] = b.report["calibration"]
    sets = build_sets(random.Random(2024))
    det = {}
    for mode in ["keyword", "novelty", "confidence", "calibrated"]:
        cli = Cli(RUN / "rt", trust)
        det[mode] = {}
        for sname, rows in sets.items():
            f = RUN / f"{sname}.jsonl"; write(f, rows)
            out = cli._run_detect(mode, "batch", "--cases", str(f))
            det[mode][sname] = metrics(sname, rows, out["results"])
        print(mode, {k: (round(v.get("fabrication_rate", -1), 3) if "fabrication_rate" in v else round(v.get("accuracy_known_only") or 0, 3)) for k, v in det[mode].items()})
    report["detection_ablation"] = det
    # ---- escalation scenario over a task stream ----
    rt2 = Cli(RUN / "rt2", trust); teacher2 = TeacherSimulator(); svc2 = BuildService(RUN / "keys", "build-svc-1", RUN / "build2", teacher2)
    esc = Escalator(rt2, teacher2, svc2, RUN / "esc")
    rng = random.Random(5)
    stream = []
    for t in DIM:   # first encounters, then related re-encounters
        ex, _ = splits(t)["eval"]
        stream.append((CANON[t], ex[0], "first"))
    for rep in range(3):
        for t in DIM:
            ex, _ = splits(t)["eval"]
            stream.append((PARA[t][1 + rep % 3] if rep else CANON[t], ex[1 + rep], "related"))
    stream += [("translate this sentence to french", [0.2, 0.7], "unknown"), ("what is the weather today", [0.2, 0.7], "research"),
               ("send email to my colleague", [0.2, 0.7], "tool"), ("remember where my car is parked", [0.2, 0.7], "memory")]
    for t in DIM:
        ex, _ = splits(t)["eval"]
        stream.append((CANON[t], ex[9], "related"))
    stream_log = []
    for intent, x, kind in stream:
        before = esc.teacher_calls
        r = esc.solve(intent, x)
        stream_log.append({"kind": kind, "intent": intent, "result": r["result"], "path": r["path"], "teacher_called": esc.teacher_calls > before, "status": r.get("status")})
    esc.online = False
    off = esc.solve("classify the weather type", [0.1, 0.2])    # unknown while offline
    off_known = esc.solve(CANON["compare_numbers"], splits("compare_numbers")["eval"][0][3])   # known capability still works offline
    calls_before = esc.teacher_calls
    report["escalation"] = {
        "stream": stream_log, "teacher_calls_total": esc.teacher_calls, "tasks": len(stream_log),
        "teacher_call_rate_first_half": sum(s["teacher_called"] for s in stream_log[:len(stream_log)//2]) / (len(stream_log)//2),
        "teacher_call_rate_second_half": sum(s["teacher_called"] for s in stream_log[len(stream_log)//2:]) / (len(stream_log) - len(stream_log)//2),
        "related_encounters_local": sum(1 for s in stream_log if s["kind"] == "related" and not s["teacher_called"] and s["result"] == "ANSWER"),
        "related_encounters_total": sum(1 for s in stream_log if s["kind"] == "related"),
        "offline_unknown": {"result": off["result"], "teacher_called_while_offline": esc.teacher_calls != calls_before},
        "offline_known_still_answers": off_known["result"] == "ANSWER",
        "pending_queue_lines": len((RUN / "esc/pending_help.jsonl").read_text().splitlines()),
        "memory_entries": len(esc.memory)}
    # malformed / hostile teacher output must never activate anything
    bad = {}
    for label, text in {"not_json": "I think you should learn it!", "unknown_action": json.dumps({"action": "rm_rf"}),
                        "bad_learning_package": json.dumps({"action": "new_capability", "learning_package": {"x": 1}}),
                        "reroute_to_missing": json.dumps({"action": "reroute", "reroute_intent": "x", "capability_id": "ghost"})}.items():
        rt3 = Cli(RUN / f"rt_bad_{label}", trust); e = Escalator(rt3, Garbage(text), svc2, RUN / f"esc_bad_{label}")
        r = e.solve("compare these two numbers", [0.1, 0.9])
        bad[label] = {"result": r["result"], "capabilities_after": len(rt3.list())}
    report["hostile_teacher_outputs"] = bad
    # privacy: what actually leaves the device
    secret_intent = "compare numbers for user bob@example.com password=hunter2 token sk-abcdef1234567890 in /home/bob/secret/file.txt"
    from integrations.teacher.provider import make_help_request
    rq = make_help_request(secret_intent, 2, ["compare_numbers"], {"reason_code": "NO_MATCH"}, "")
    sent = rq.to_json()
    report["privacy"] = {"sent_request": json.loads(sent), "leaks": [s for s in ["bob@example.com", "hunter2", "sk-abcdef1234567890", "/home/bob"] if s in sent],
                         "contains_input_values": False}
    # information classification
    labelled = [("My car is on level 3", "MEMORY"), ("I left my keys on the kitchen table", "MEMORY"), ("Remember that my meeting is at 5pm", "MEMORY"),
                ("We parked on level 2", "MEMORY"), ("My sister lives in Leeds", "MEMORY"), ("I put the passport in the blue folder", "MEMORY"),
                ("This API uses OAuth", "KNOWLEDGE"), ("The library requires version 3 or later", "KNOWLEDGE"), ("The endpoint returns JSON by default", "KNOWLEDGE"),
                ("The tool supports the CSV format", "KNOWLEDGE"), ("Paris is the capital of France", "KNOWLEDGE"), ("Water boils at 100 degrees Celsius", "KNOWLEDGE"),
                ("Learn to recognize a new visual pattern", "CAPABILITY"), ("Learn to compare two numbers", "CAPABILITY"), ("Be able to classify points inside a region", "CAPABILITY"),
                ("Get better at detecting spam", "CAPABILITY"), ("Learn how to sort these items", "CAPABILITY"), ("Be able to predict the next value", "CAPABILITY"),
                ("Identify which of four values is largest", "CAPABILITY"), ("Distinguish cats from dogs in photos", "CAPABILITY")]
    preds = [(t, y, classify_information(t)) for t, y in labelled]
    report["information_classification"] = {"n": len(preds), "accuracy": sum(1 for _, y, p in preds if y == p) / len(preds),
                                            "errors": [(t, y, p) for t, y, p in preds if y != p],
                                            "note": "tiny hand-written rule baseline on a hand-written set; not evidence of general classification ability"}
    tricky = [("Teach yourself to tell sarcasm from sincerity", "CAPABILITY"), ("The sort function in this library is stable", "KNOWLEDGE"),
              ("Where did I put my glasses", "MEMORY"), ("Improve at spotting fraudulent invoices", "CAPABILITY"),
              ("The default timeout is 30 seconds", "KNOWLEDGE"), ("I need the tool to predict churn", "CAPABILITY"),
              ("Our API supports sorting by date", "KNOWLEDGE"), ("My laptop is in the car", "MEMORY"),
              ("Learn which emails are urgent", "CAPABILITY"), ("The classifier endpoint returns a label", "KNOWLEDGE")]
    tp = [(t, y, classify_information(t)) for t, y in tricky]
    report["information_classification"]["tricky_set"] = {"n": len(tp), "accuracy": sum(1 for _, y, p in tp if y == p) / len(tp),
                                                          "errors": [(t, y, p) for t, y, p in tp if y != p]}
    out = ROOT / "benchmarks/reports/phase5.json"; out.write_text(json.dumps(report, indent=1, default=str)); print("wrote", out)


if __name__ == "__main__":
    main()
