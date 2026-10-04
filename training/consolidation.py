"""Consolidation: duplicate/overlap detection, guarded merge, distillation, structured pruning, archive policy.
Every operation is gated by regression tests and is reversible (versions are kept; archived capabilities are restorable)."""
from __future__ import annotations
import copy, json, zipfile
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from training.trainer.train import accuracy, make_mlp, count_params

DUP_AGREEMENT = 0.98


def _next_version(cap):
    a, b, c = map(int, cap["active_version"].split("."))
    return f"{a}.{b}.{c + 1}"


def probe_inputs(stats_list, n=1500, seed=0):
    lo = np.min([s["min"] for s in stats_list], 0); hi = np.max([s["max"] for s in stats_list], 0)
    return np.random.RandomState(seed).uniform(lo, hi, size=(n, len(lo))).tolist()


def predictions(cli, cap_id, xs, workdir: Path):
    f = Path(workdir) / f"probe_{cap_id}.jsonl"
    f.write_text("\n".join(json.dumps({"intent": "x", "input": x}) for x in xs) + "\n")
    out = cli._run_detect("keyword", "batch", "--cases", str(f), capability=cap_id)
    return [r["label_index"] for r in out["results"]]


def pairwise_report(cli, workdir: Path, n=1500):
    """Functional agreement of every comparable pair, measured two ways:
    - in-distribution: on the union of both packages' bundled test inputs (the domain both were trained for);
    - uniform box: on random inputs inside the union of their training ranges (includes regions neither module was trained on)."""
    caps = [c for c in cli.list() if c.get("input_stats")]
    rows = []
    for i, a in enumerate(caps):
        for b in caps[i + 1:]:
            if a["input_dim"] != b["input_dim"] or a["labels"] != b["labels"]:
                continue
            xs_in = [t["input"] for t in package_tests(cli, a) + package_tests(cli, b)]
            xs_box = probe_inputs([a["input_stats"], b["input_stats"]], n)
            ag = {}
            for tag, xs in (("in_distribution", xs_in), ("uniform_box", xs_box)):
                pa, pb = predictions(cli, a["capability_id"], xs, workdir), predictions(cli, b["capability_id"], xs, workdir)
                ag[tag] = float(np.mean(np.array(pa) == np.array(pb)))
            agree = ag["in_distribution"]
            rows.append({"a": a["capability_id"], "b": b["capability_id"], "agreement_in_distribution": ag["in_distribution"], "agreement_uniform_box": ag["uniform_box"],
                         "verdict": "duplicate" if agree >= DUP_AGREEMENT else "overlapping_different_function" if agree >= 0.6 else "different"})
    return rows


def package_tests(cli, cap):
    z = zipfile.ZipFile(cli.root / "store" / cap["store_file"])
    return [json.loads(l) for l in z.read("tests/cases.jsonl").decode().splitlines() if l.strip()]


def accuracy_via_runtime(cli, cap_id, cases, workdir: Path):
    f = Path(workdir) / f"verify_{cap_id}.jsonl"
    f.write_text("\n".join(json.dumps({"intent": "x", "input": c["input"], "expected_index": c["expected"]}) for c in cases) + "\n")
    return cli._run_detect("keyword", "batch", "--cases", str(f), capability=cap_id)["summary"]["accuracy"]


def merge_duplicate(cli, svc, keeper: str, dup: str, routing_suite, workdir: Path) -> dict:
    """Make `dup` redundant: keeper must pass dup's bundled tests, inherit its routing keywords (verified not to break known routes);
    then dup is archived (kept in store, restorable)."""
    log = {"keeper": keeper, "duplicate": dup, "steps": []}
    lst = {c["capability_id"]: c for c in cli.list()}
    acc = accuracy_via_runtime(cli, keeper, package_tests(cli, lst[dup]), workdir)
    log["steps"].append({"step": "keeper_passes_duplicate_tests", "accuracy": acc})
    if acc < 0.95:
        log["merged"] = False; return log
    built = svc.repackage(keeper, _next_version(lst[keeper]), extra_keywords=lst[dup]["keywords"], strategy="merge_duplicate", note=f"absorbed {dup}")
    imp = cli.import_caps(str(built.cap_path))[0]
    log["steps"].append({"step": "import_keeper_with_merged_keywords", "activated": imp["activated"]})
    if not imp["activated"]:
        log["merged"] = False; return log
    cli.json("archive", dup)          # tentatively remove the duplicate from routing, then check that every known route still works
    bad = [i for i, x, c in routing_suite if (r := cli.solve(i, x)).get("capability_id") != (keeper if c == dup else c) or r.get("status") != "KNOWN"]
    log["steps"].append({"step": "routing_regression_after_archiving_duplicate", "changed_routes": bad})
    if bad:
        cli.json("restore", dup) if False else cli._run("restore", dup)
        cli.json("rollback", keeper); log["merged"] = False; log["rolled_back"] = True; return log
    log["merged"] = True
    return log


def _train_student(student, xs, ys, epochs, lr=5e-3):
    xt, yt = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys)
    opt = torch.optim.Adam(student.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        student.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = perm[i:i + 128]; opt.zero_grad(); lossf(student(xt[b]), yt[b]).backward(); opt.step()
    return student


def _teacher_labels(model, xs):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xs, dtype=torch.float32)).argmax(1).tolist()


def distill(svc, cap_id, hidden=(16, 16), n_probe=8000, epochs=120, seed=0):
    m = svc.models[cap_id]; torch.manual_seed(seed)
    teacher = m["model"]
    xs = probe_inputs([m["stats"]], n_probe, seed) + m["replay"][0]
    ys = _teacher_labels(teacher, xs)
    student = _train_student(make_mlp(m["input_dim"], len(m["labels"]), hidden), xs, ys, epochs)
    return student, {"strategy": "distillation", "hidden": list(hidden), "teacher_heldout": accuracy(teacher, *m["heldout"]),
                     "student_heldout": accuracy(student, *m["heldout"]), "params_before": count_params(teacher), "params_after": count_params(student)}


def prune_structured(svc, cap_id, keep=0.5, finetune_epochs=60, seed=0):
    """Remove the hidden units with smallest weight-norm in each hidden layer, then fine-tune on teacher labels."""
    m = svc.models[cap_id]; torch.manual_seed(seed)
    lin = [l for l in m["model"] if isinstance(l, nn.Linear)]
    keeps, prev = [], torch.arange(lin[0].in_features)
    new = []
    for li, l in enumerate(lin):
        W, b = l.weight.data[:, prev], l.bias.data
        if li < len(lin) - 1:
            score = W.abs().sum(1) + (lin[li + 1].weight.data.abs().sum(0))
            k = max(2, int(round(W.shape[0] * keep))); idx = torch.topk(score, k).indices.sort().values
            new.append((W[idx], b[idx])); prev = idx
        else:
            new.append((W, b))
    layers = []
    for li, (W, b) in enumerate(new):
        ln = nn.Linear(W.shape[1], W.shape[0]); ln.weight.data, ln.bias.data = W.clone(), b.clone(); layers.append(ln)
        if li < len(new) - 1:
            layers.append(nn.ReLU())
    pruned = nn.Sequential(*layers)
    xs = probe_inputs([m["stats"]], 6000, seed) + m["replay"][0]
    pruned = _train_student(pruned, xs, _teacher_labels(m["model"], xs), finetune_epochs, lr=2e-3)
    return pruned, {"strategy": "structured_pruning", "keep": keep, "teacher_heldout": accuracy(m["model"], *m["heldout"]),
                    "student_heldout": accuracy(pruned, *m["heldout"]), "params_before": count_params(m["model"]), "params_after": count_params(pruned)}


def compact(svc, cli, cap_id, student, report, max_drop=0.01, workdir: Path | None = None) -> dict:
    """Install a distilled/pruned model if it stays within `max_drop` of the original on held-out data and passes the bundled tests."""
    out = {**report, "installed": False}
    if report["student_heldout"] < report["teacher_heldout"] - max_drop:
        out["reason"] = f"accuracy drop {report['teacher_heldout'] - report['student_heldout']:.3f} > {max_drop}"; return out
    cur = next(c for c in cli.list() if c["capability_id"] == cap_id)
    b = svc.swap_model(cap_id, student, _next_version(cur), report["strategy"])
    imp = cli.import_caps(str(b.cap_path))[0]
    out["activated"] = imp["activated"]; out["failed_step"] = imp.get("failed_step")
    if imp["activated"]:
        svc.commit(); out["installed"] = True; out["bytes_after"] = next(c for c in cli.list() if c["capability_id"] == cap_id)["model_bytes"]
    return out


def archive_unused(cli, min_calls=1):
    """Archive capabilities with fewer than `min_calls` recorded uses (never if another active capability depends on them)."""
    done, refused = [], []
    for c in cli.json("list-all"):
        if c["role"] == "capability" and not c["archived"] and c["calls"] < min_calls:
            p = cli._run("archive", c["capability_id"], check=False)
            (done if p.returncode == 0 else refused).append({"capability_id": c["capability_id"], "stderr": p.stderr.strip()[:120]})
    return {"archived": done, "refused": refused}
