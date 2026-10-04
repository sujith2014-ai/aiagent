"""Learned router: hashed word/char-trigram features -> MLP -> {capability ids..., UNKNOWN}.
The feature hashing must stay bit-identical to core/src/router.rs (`hash_features`); tests compare both."""
from __future__ import annotations
import re, time, copy
import numpy as np, torch, torch.nn as nn

DIM = 256


def fnv1a(b: bytes) -> int:
    h = 0x811C9DC5
    for x in b:
        h ^= x
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def hash_features(text: str, dim: int = DIM) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    for w in [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]:
        v[fnv1a(f"w:{w}".encode()) % dim] += 1.0
        padded = f"#{w}#".encode()
        for i in range(len(padded) - 2):
            v[fnv1a(b"t:" + padded[i:i + 3]) % dim] += 1.0
    n = float(np.sqrt((v * v).sum()))
    if n > 0:
        v /= n
    return v.tolist()


def train_router(examples: list[tuple[str, str]], classes: list[str], dim=DIM, hidden=64, epochs=300, seed=0, lr=5e-3):
    """examples: (intent text, class name); classes includes "UNKNOWN" last. Returns (model, report)."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    idx = rng.permutation(len(examples)); cut = int(len(idx) * 0.85)
    tr, va = [examples[i] for i in idx[:cut]], [examples[i] for i in idx[cut:]]
    X = lambda ex: torch.tensor([hash_features(t, dim) for t, _ in ex], dtype=torch.float32)
    Y = lambda ex: torch.tensor([classes.index(c) for _, c in ex])
    xt, yt, xv, yv = X(tr), Y(tr), X(va), Y(va)
    model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, len(classes)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); lossf = nn.CrossEntropyLoss()
    t0 = time.time(); best, bs = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lossf(model(xt[b]), yt[b]).backward(); opt.step()
        with torch.no_grad():
            va_acc = (model(xv).argmax(1) == yv).float().mean().item()
        if va_acc >= best:
            best, bs = va_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(bs)
    return model, {"train_seconds": time.time() - t0, "val_accuracy": best, "n_train": len(tr), "n_val": len(va),
                   "params": sum(p.numel() for p in model.parameters()), "classes": classes}
