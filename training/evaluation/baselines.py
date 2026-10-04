"""Conventional tiny-network baselines for the same three tasks.
 - joint:      one shared MLP trained on all tasks at once (upper bound for a single network)
 - sequential: one shared MLP fine-tuned on A, then B, then C (measures catastrophic forgetting)"""
from __future__ import annotations
import torch, torch.nn as nn
from training.trainer.train import make_mlp, count_params, torch
from integrations.teacher.simulator import TASKS, sample, splits

NAMES = list(TASKS)
DIN = max(t["dim"] for t in TASKS.values()) + len(TASKS)
DOUT = max(len(t["labels"]) for t in TASKS.values())


def encode(task: str, x):
    t = NAMES.index(task)
    v = list(x) + [0.0] * (max(tt["dim"] for tt in TASKS.values()) - len(x)) + [1.0 if i == t else 0.0 for i in range(len(NAMES))]
    return v


def data(task: str, n, seed, exclude=None, unique=False):
    xs, ys = sample(task, n, seed, exclude=exclude, unique=unique)
    return torch.tensor([encode(task, x) for x in xs], dtype=torch.float32), torch.tensor(ys), len(TASKS[task]["labels"])


def _acc(model, task, ds):
    x, y, k = ds
    with torch.no_grad():
        return (model(x)[:, :k].argmax(1) == y).float().mean().item()


def _fit(model, sets, epochs=150, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xs = torch.cat([s[0] for s in sets]); ys = torch.cat([s[1] for s in sets])
    ks = torch.cat([torch.full((len(s[1]),), s[2]) for s in sets])
    for _ in range(epochs):
        perm = torch.randperm(len(xs))
        for i in range(0, len(xs), 128):
            idx = perm[i:i + 128]
            logits = model(xs[idx]).clone()
            mask = torch.arange(DOUT)[None, :] >= ks[idx][:, None]
            logits[mask] = -1e9
            opt.zero_grad(); nn.functional.cross_entropy(logits, ys[idx]).backward(); opt.step()


def run(hidden=(48, 48), seed=0):
    torch.manual_seed(seed)
    train, test = {}, {}
    for n in NAMES:
        sp = splits(n)
        enc = lambda xy: (torch.tensor([encode(n, x) for x in xy[0]], dtype=torch.float32), torch.tensor(xy[1]), len(TASKS[n]["labels"]))
        train[n], test[n] = enc(sp["train"]), enc(sp["eval"])
    out = {"hidden": list(hidden), "params": count_params(make_mlp(DIN, DOUT, hidden))}
    joint = make_mlp(DIN, DOUT, hidden); _fit(joint, list(train.values()))
    out["joint"] = {n: _acc(joint, n, test[n]) for n in NAMES}
    seq = make_mlp(DIN, DOUT, hidden); stages = []
    for i, n in enumerate(NAMES):
        _fit(seq, [train[n]])
        stages.append({"after": n, "acc": {m: _acc(seq, m, test[m]) for m in NAMES[:i + 1]}})
    out["sequential"] = stages
    return out
