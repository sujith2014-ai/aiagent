"""Phase 9: real-world controlled tasks. Four real datasets (bundled with scikit-learn) are taught by example through the desktop app
(no teacher), then evaluated through the Rust runtime and compared against conventional baselines on identical splits."""
from __future__ import annotations
import json, math, shutil, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.cli import Cli, ROOT
from apps.desktop.app import DesktopApp, init
from training.learning_package.examples import package_from_examples

RUN = ROOT / "runs" / "phase9"
TASKS = [("iris_species", load_iris, "classify iris flower species from sepal and petal measurements"),
         ("wine_cultivar", load_wine, "identify wine cultivar from chemical analysis"),
         ("digit_recognition", load_digits, "recognize handwritten digit from 8x8 pixel intensities"),
         ("tumor_malignancy", load_breast_cancer, "classify breast tumor as malignant or benign from cell measurements")]


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def ece_conf(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float); edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def write_cases(path, intent, xs, ys):
    Path(path).write_text("\n".join(json.dumps({"intent": intent, "input": x, "expected_index": int(y)}) for x, y in zip(xs, ys)) + "\n")


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    app = DesktopApp(init(RUN / "state"))
    rep = {"benchmark_format": "bench/1", "experiment": "phase9-real-world-tasks", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"),
           "datasets": "scikit-learn bundled copies of iris, wine, digits, breast cancer (duplicates removed, stratified 60/20/20 split, seed 0; baselines fit on the same 60% train split)",
           "tasks": {}}
    prev = {}
    for cap, loader, intent in TASKS:
        ds = loader(); xs, ys = ds.data.tolist(), ds.target.tolist()
        labels = [str(t) for t in ds.target_names]
        lp, (xte, yte), info = package_from_examples(cap, intent, ["x"], labels, xs, ys, seed=0)
        xtr, ytr = np.array(lp.train_x), np.array(lp.train_y)
        base = {}
        for name, mdl in (("majority", None), ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                          ("knn_k5", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                          ("sklearn_mlp_32x32", make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=1500, random_state=0)))):
            if mdl is None:
                maj = int(np.bincount(ytr).argmax()); acc = float(np.mean(np.array(yte) == maj)); params = 0
            else:
                mdl.fit(xtr, ytr); acc = float(mdl.score(xte, yte))
                if name == "logistic_regression":
                    params = xtr.shape[1] * (len(labels) if len(labels) > 2 else 1) + (len(labels) if len(labels) > 2 else 1)
                elif name == "knn_k5":
                    params = int(xtr.size)                       # a k-NN model stores the training set
                else:
                    params = int(sum(c.size for c in mdl[-1].coefs_) + sum(b.size for b in mdl[-1].intercepts_))
            base[name] = {"accuracy": acc, "params_or_stored_floats": params}
        gate = float(min(0.95, max(0.90, base["logistic_regression"]["accuracy"] - 0.03)))
        t0 = time.time()
        out = app.teach({"capability_id": cap, "description": intent, "labels": labels, "x": xs, "y": ys, "min_acc": gate, "seed": 0})
        secs = time.time() - t0
        entry = {"data": info, "promotion_gate": gate, "baselines": base, "teach": {k: out.get(k) for k in ("learned", "reason", "version", "heldout_accuracy", "params", "teacher_used")}, "teach_seconds": secs}
        if out.get("learned"):
            cf = RUN / f"test_{cap}.jsonl"; write_cases(cf, intent, xte, yte)
            b = app.cli.batch(str(cf))
            ans = [(r, y) for r, y in zip(b["results"], yte) if not r.get("needs_help")]
            ok_ans = [1 if r["label_index"] == y else 0 for r, y in ans]
            acc = float(sum(ok_ans)) / len(yte)                   # overall: refusals count as misses
            entry["runtime"] = {"accuracy_overall": acc, "ci95_overall": wilson(acc, len(yte)), "coverage": len(ans) / len(yte), "n_test": len(yte),
                                "accuracy_answered": float(np.mean(ok_ans)) if ok_ans else None, "refused_as_out_of_range": len(yte) - len(ans),
                                "latency_us_p50": b["summary"]["latency_us_p50"], "params": out["params"],
                                "model_bytes": next(c["model_bytes"] for c in app.cli.list() if c["capability_id"] == cap),
                                "ece_calibrated_answered": ece_conf([r["confidence"] for r, _ in ans], ok_ans) if ans else None,
                                "ece_raw_answered": ece_conf([r["raw_confidence"] for r, _ in ans], ok_ans) if ans else None}
            prev[cap] = (intent, xte, yte, acc)
            entry["accuracy_of_all_earlier_after_this_stage"] = {c: app.cli.batch(str(RUN / f"test_{c}.jsonl"))["summary"]["accuracy"] for c in prev}
        rep["tasks"][cap] = entry
        print(cap, "gate", round(gate, 3), "learned", out.get("learned"), "runtime acc", entry.get("runtime", {}).get("accuracy_overall"), "coverage", entry.get("runtime", {}).get("coverage"), "LR", round(base["logistic_regression"]["accuracy"], 3))
    # forgetting across the 4 sequential stages
    drops = []
    for cap, e in rep["tasks"].items():
        first = e["runtime"]["accuracy_overall"]
        for later, e2 in rep["tasks"].items():
            if cap in e2.get("accuracy_of_all_earlier_after_this_stage", {}):
                drops.append(first - e2["accuracy_of_all_earlier_after_this_stage"][cap])
    rep["max_forgetting_drop"] = max(drops) if drops else None
    # routing with all four installed
    route = {}
    for cap, (intent, xte, yte, _) in prev.items():
        correct_known = 0
        for x in xte[:40]:
            r = app.cli.solve(intent, x)
            correct_known += int(r.get("result") == "ANSWER" and r["capability_id"] == cap and r["status"] in ("KNOWN", "UNCERTAIN"))
        route[cap] = correct_known / len(xte[:40])
    rep["routing_correct_capability_fraction"] = route
    # out-of-range inputs
    ood = {}
    for cap, (intent, xte, yte, _) in prev.items():
        k = 0
        for x in xte[:30]:
            r = app.cli.solve(intent, [v * 10 + 1000 for v in x]); k += int(r.get("result") == "NEEDS_HELP" and r.get("reason_code") == "OUT_OF_DISTRIBUTION")
        ood[cap] = k / len(xte[:30])
    rep["out_of_range_refused_fraction"] = ood
    # novelty rule effect on the real tasks (through the Rust runtime): coverage on held-out data and detection of a +4 sd shift on every feature
    nov = {}
    for cap, (intent, xte, yte, _) in prev.items():
        ds = next(l for c, l, i in TASKS if c == cap)(); sd = np.asarray(ds.data).std(0)
        shifted = RUN / f"shift_{cap}.jsonl"; write_cases(shifted, intent, (np.asarray(xte) + 4 * sd).tolist(), yte)
        nov[cap] = {}
        for rule in ("strict", "balanced"):
            held = app.cli._run_detect("full", "batch", "--cases", str(RUN / f"test_{cap}.jsonl"), novelty=rule)["summary"]
            shf = app.cli._run_detect("full", "batch", "--cases", str(shifted), novelty=rule)["summary"]
            nov[cap][rule] = {"heldout_coverage": 1 - held["needs_help"] / held["cases"], "shift_4sd_refused": shf["needs_help"] / shf["cases"]}
    rep["novelty_rule_on_real_tasks"] = nov
    # clean restart + constrained device: same .cap files, same outputs
    exp = RUN / "export"; exp.mkdir()
    for f in (Path(app.cfg["root"]) / "store").glob("*.cap"):
        shutil.copy(f, exp / f.name)
    trust = Path(app.cfg["trust"])
    full_probs = {cap: [r.get("probs") for r in app.cli.batch(str(RUN / f"test_{cap}.jsonl"))["results"]] for cap in prev}   # None for refused rows
    shutil.rmtree(Path(app.cfg["root"]))
    fresh = Cli(RUN / "fresh", trust); reps = fresh.import_caps(*sorted(map(str, exp.glob("*.cap"))))
    prefix = ["bash", "-c", 'ulimit -v 1048576; exec taskset -c 0 "$@"', "_"] if shutil.which("taskset") else None
    cons = Cli(RUN / "cons", trust, device="PC_CONSTRAINED", prefix=prefix); creps = cons.import_caps(*sorted(map(str, exp.glob("*.cap"))))
    rs = {"fresh_import_all_activated": all(r["activated"] for r in reps), "constrained_import_all_activated": all(r["activated"] for r in creps), "per_capability": {}}
    for cap in prev:
        bf = fresh.batch(str(RUN / f"test_{cap}.jsonl")); bc = cons.batch(str(RUN / f"test_{cap}.jsonl"))
        rs["per_capability"][cap] = {"fresh_identical_to_before": [r.get("probs") for r in bf["results"]] == full_probs[cap],
                                     "constrained_max_prob_diff": max([abs(a - b) for x, y in zip(bf["results"], bc["results"]) if x.get("probs") and y.get("probs") for a, b in zip(x["probs"], y["probs"])] or [0.0]),
                                     "same_refusals_constrained": [bool(x.get("needs_help")) for x in bf["results"]] == [bool(y.get("needs_help")) for y in bc["results"]],
                                     "constrained_latency_us_p50": bc["summary"]["latency_us_p50"], "constrained_peak_rss_kb": bc["summary"]["peak_rss_kb"]}
    rep["restart_and_constrained"] = rs
    out = ROOT / "benchmarks/reports/phase9.json"; out.write_text(json.dumps(rep, indent=1, default=str)); print("wrote", out)


if __name__ == "__main__":
    main()
