"""Candidate training. Only touches the new module's own parameters (existing modules are separate
frozen artifacts in the runtime). Also hosts the conventional-baseline trainers."""
from __future__ import annotations
import time, copy
import torch, torch.nn as nn
from training.learning_package.schema import LearningPackage

torch.set_num_threads(2)


def make_mlp(din: int, dout: int, hidden=(32, 32)) -> nn.Sequential:
    layers, d = [], din
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _tensors(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def accuracy(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        xt, yt = _tensors(x, y)
        return (model(xt).argmax(1) == yt).float().mean().item()


def train_candidate(lp: LearningPackage, hidden=(32, 32), epochs=400, lr=5e-3, seed=0, patience=60):
    """Train on lp.train (+edge cases), early-stop on lp.validation. Returns (model, report)."""
    torch.manual_seed(seed)
    x = lp.train_x + lp.edge_cases_x
    y = lp.train_y + lp.edge_cases_y
    xt, yt = _tensors(x, y)
    model = make_mlp(lp.input_dim, len(lp.labels), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lossf(model(xt[idx]), yt[idx]).backward()
            opt.step()
        va = accuracy(model, lp.validation_x, lp.validation_y)
        if va > best + 1e-9:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"epochs": ep + 1, "val_accuracy": best, "train_accuracy": accuracy(model, x, y),
                   "train_seconds": time.time() - t0, "params": count_params(model), "hidden": list(hidden), "seed": seed}
