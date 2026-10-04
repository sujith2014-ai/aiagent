"""Phase 12: on-device learning vs server training; few-shot adaptation under input drift. Measured on this PC; no phone, battery, thermal or NPU data (PENDING, R4)."""
from __future__ import annotations
import json, shutil, sys, time, platform, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from scripts.phase11_lab import Lab
from scripts.phase12_lab import Device
from scripts.cli import ROOT
from packages.capbuild import keygen, write_trust

RUN = ROOT / "runs" / "phase12"
TASKS = [("iris_species", load_iris, "classify iris flower species from measurements"), ("wine_cultivar", load_wine, "identify wine cultivar from chemical analysis"),
         ("digit_recognition", load_digits, "recognize handwritten digit from pixel intensities"), ("tumor_malignancy", load_breast_cancer, "classify breast tumor malignant or benign from cell measurements")]
# the 'phone-like' constraint: one core, 256 MB address space
CONSTRAIN = ["taskset", "-c", "0", "prlimit", "--as=268435456"]
DRAWS, NS = 5, (10, 25)
DRIFT_STD = 1.5


def eval_cap(cli, cap, x, y, intent):
    f = RUN / "eval.jsonl"; f.write_text("\n".join(json.dumps({"intent": intent, "input": [float(v) for v in r], "expected_index": int(c)}) for r, c in zip(x, y)) + "\n")
    res = cli._run_detect("keyword", "batch", "--cases", str(f), capability=cap)["results"]
    return float(np.mean([r.get("label_index") == int(c) for r, c in zip(res, y)]))


def drift_cols(x):
    return np.argsort(x.var(axis=0))[-2:]


def shifted(x, cols, sd):
    z = x.copy(); z[:, cols] += DRIFT_STD * sd[cols]; return z


def split(n, seed):
    p = np.random.default_rng(seed).permutation(n); a, b = int(.5 * n), int(.75 * n); return p[:a], p[a:b], p[b:]


def spec_for(cap, intent, labels, x, y, **kw):
    return {"capability_id": cap, "description": intent, "keywords": [w for w in (intent + " " + cap.replace("_", " ")).lower().split() if len(w) > 2], "labels": labels, "x": x.tolist(), "y": [int(v) for v in y], "min_accuracy": 0.0, **kw}


def main():
    if RUN.exists(): shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keygen("server-build-1", RUN / "srvkeys"); write_trust(RUN / "trust.json", {"server-build-1": (RUN / "srvkeys/server-build-1.public").read_text()})
    rep = {"benchmark_format": "bench/1", "experiment": "phase12-on-device-learning", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"), "constraint": "taskset -c 0 + RLIMIT_AS 256MB for on-device runs",
           "not_measured": ["battery", "thermal throttling", "NPU/GPU", "real phone CPU (this is a PC core pinned to one CPU)"]}
    L = Lab(RUN / "lab", device="PC_CONSTRAINED")
    # ---- A. training from scratch: device (Rust, constrained) vs server (PyTorch) on the same 70% split, scored on the same 30% ----
    A = {}
    for cap, loader, intent in TASKS:
        d = loader(); labels = [str(t) for t in d.target_names]; runs = []
        for seed in range(DRAWS):
            p = np.random.default_rng(seed).permutation(len(d.data)); k = int(.7 * len(p)); tr, te = p[:k], p[k:]
            dev = Device(RUN / f"A_{cap}_{seed}", RUN / "trust.json", prefix=CONSTRAIN)
            t0 = time.time(); r = dev.learn(spec_for(cap, intent, labels, d.data[tr], d.target[tr], min_accuracy=0.0, seed=seed)); wall = time.time() - t0
            dev_acc = eval_cap(dev.cli, cap, d.data[te], d.target[te], intent) if r["learned"] else None
            sid = f"{cap}_a{seed}"                                    # server: the real build service on the same split; scored with the same evaluator on the same rows
            t0 = time.time(); sr = L.client._http("POST", "/train", {"capability_id": sid, "description": intent, "labels": labels, "x": d.data[tr].tolist(), "y": d.target[tr].tolist(), "min_acc": 0.0, "seed": seed}); st = time.time() - t0
            L.client.sync(); srv_acc = eval_cap(L.local, sid, d.data[te], d.target[te], intent) if sr["learned"] else None
            spkg = next((q for q in L.app.packages.values() if q["capability_id"] == sid), None)
            runs.append({"device_acc": dev_acc, "server_acc": srv_acc, "device_train_s": r["train_seconds"], "device_wall_s": wall, "device_rss_mb": r["peak_rss_kb"] / 1024, "device_bytes": r["package_bytes"], "device_params": r["params"],
                         "server_s": st, "server_bytes": spkg["size"] if spkg else None, "server_reason": sr.get("reason")})
        def m(key): v = [r[key] for r in runs if r[key] is not None]; return round(float(np.mean(v)), 4) if v else None
        def sdv(key): v = [r[key] for r in runs if r[key] is not None]; return round(float(np.std(v)), 4) if v else None
        A[cap] = {"train_examples": int(.7 * len(d.data)), "eval_examples": len(d.data) - int(.7 * len(d.data)), "splits": len(runs),
                  "device": {"accuracy_mean": m("device_acc"), "accuracy_std": sdv("device_acc"), "train_s_mean": m("device_train_s"), "wall_s_mean_incl_process_and_pack": m("device_wall_s"), "peak_rss_mb_max": round(max(r["device_rss_mb"] for r in runs), 1), "package_bytes_mean": m("device_bytes"), "params": runs[0]["device_params"]},
                  "server": {"accuracy_mean": m("server_acc"), "accuracy_std": sdv("server_acc"), "train_and_sign_s_mean": m("server_s"), "package_bytes_mean": m("server_bytes"), "failures": [r["server_reason"] for r in runs if r["server_reason"]]},
                  "per_split_accuracy": [[r["device_acc"], r["server_acc"]] for r in runs]}
    rep["from_scratch"] = A
    rep["server_process_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    # ---- B. few-shot adaptation to an input drift (two highest-variance features read 1.5 sd high) ----
    B = {}
    for cap, loader, intent in TASKS:
        d = loader(); X, Y = d.data, d.target; labels = [str(t) for t in d.target_names]; cols = drift_cols(X); sd = X.std(axis=0)
        B[cap] = {"drift_columns": [int(c) for c in cols], "n": {}, "new_module_refusals": {}}
        for n in NS:
            refused = []; server_failures = []
            acc = {m: {"new": [], "old": [], "secs": [], "accepted_by_default_gates": []} for m in ("frozen", "head", "full", "new_small_module", "server_retrain")}
            for seed in range(DRAWS):
                a, b, c = split(len(X), seed)
                dev = Device(RUN / f"B_{cap}_{n}_{seed}", RUN / "trust.json", prefix=CONSTRAIN)
                base = dev.learn(spec_for(cap, intent, labels, X[a], Y[a], seed=seed))
                if not base["learned"]: continue
                rng = np.random.default_rng(100 + seed); pick = rng.choice(b, size=min(n, len(b)), replace=False)
                xn, yn = shifted(X[pick], cols, sd), Y[pick]; xnew_te, ynew_te = shifted(X[c], cols, sd), Y[c]; xold_te, yold_te = X[c], Y[c]
                rb = rng.choice(a, size=min(40, len(a)), replace=False)               # the small replay buffer a device could keep
                def score(cli, capid): return eval_cap(cli, capid, xnew_te, ynew_te, intent), eval_cap(cli, capid, xold_te, yold_te, intent)
                f_new, f_old = score(dev.cli, cap)
                for m, (nw, od, sec) in {"frozen": (f_new, f_old, 0.0)}.items():
                    acc[m]["new"].append(nw); acc[m]["old"].append(od); acc[m]["secs"].append(sec); acc[m]["accepted_by_default_gates"].append(False)
                for mode in ("head", "full"):
                    dv = Device(RUN / f"B_{cap}_{n}_{seed}_{mode}", RUN / "trust.json", prefix=CONSTRAIN); dv.learn(spec_for(cap, intent, labels, X[a], Y[a], seed=seed))   # same base for every mode
                    r = dv.adapt({"capability_id": cap, "mode": mode, "x": xn.tolist(), "y": [int(v) for v in yn], "x_test": xnew_te.tolist(), "y_test": [int(v) for v in ynew_te],
                                  "x_old": X[rb].tolist(), "y_old": [int(v) for v in Y[rb]], "max_old_drop": 1.0, "min_gain": -1.0, "seed": seed})
                    nw, od = score(dv.cli, cap)
                    acc[mode]["new"].append(nw); acc[mode]["old"].append(od); acc[mode]["secs"].append(r["train_seconds"])
                    acc[mode]["accepted_by_default_gates"].append(bool(nw > f_new and od >= f_old - 0.05))
                # new small module on the drifted samples only (no old knowledge)
                dn = Device(RUN / f"B_{cap}_{n}_{seed}_new", RUN / "trust.json", prefix=CONSTRAIN)
                try: r = dn.learn(spec_for(cap, intent, labels, xn, yn, seed=seed))
                except RuntimeError as e: r = {"learned": False}; refused.append(str(e).split("\n")[0])      # e.g. fewer than 30 examples: the device refuses to train a module
                if r["learned"]:
                    nw, od = score(dn.cli, cap); acc["new_small_module"]["new"].append(nw); acc["new_small_module"]["old"].append(od); acc["new_small_module"]["secs"].append(r["train_seconds"]); acc["new_small_module"]["accepted_by_default_gates"].append(bool(nw > f_new and od >= f_old - 0.05))
                # server retrain on old + new data (the server can hold the data; the device cannot)
                sid = f"{cap}_s{n}_{seed}"
                xs, ys = np.vstack([X[a], xn]), np.concatenate([Y[a], yn])
                t0 = time.time(); sr = L.client._http("POST", "/train", {"capability_id": sid, "description": intent, "labels": labels, "x": xs.tolist(), "y": ys.tolist(), "min_acc": 0.0}); st = time.time() - t0
                if not sr["learned"]: server_failures.append(sr.get("reason"))
                if sr["learned"]:
                    L.client.sync(); nw, od = score(L.local, sid); acc["server_retrain"]["new"].append(nw); acc["server_retrain"]["old"].append(od); acc["server_retrain"]["secs"].append(st); acc["server_retrain"]["accepted_by_default_gates"].append(bool(nw > f_new and od >= f_old - 0.05))
            B[cap]["new_module_refusals"][str(n)] = {"count": len(refused), "reason": refused[0] if refused else None}
            B[cap].setdefault("server_retrain_failures", {})[str(n)] = {"count": len(server_failures), "reason": server_failures[0] if server_failures else None}
            B[cap]["n"][str(n)] = {m: {"new_domain_accuracy_mean": round(float(np.mean(v["new"])), 3), "new_std": round(float(np.std(v["new"])), 3), "old_domain_accuracy_mean": round(float(np.mean(v["old"])), 3),
                                       "train_s_mean": round(float(np.mean(v["secs"])), 3), "draws": len(v["new"]), "default_gate_acceptance": round(float(np.mean(v["accepted_by_default_gates"])), 2) if v["accepted_by_default_gates"] else None} for m, v in acc.items() if v["new"]}
    rep["adaptation"] = B
    (ROOT / "benchmarks/reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "benchmarks/reports/phase12.json").write_text(json.dumps(rep, indent=1)); L.stop(); print("ok")


if __name__ == "__main__":
    main()
