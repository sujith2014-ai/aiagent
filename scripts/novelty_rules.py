"""Compare candidate out-of-distribution rules on real datasets: false refusals on legitimate held-out data vs detection of several shifts.
Rules use only the statistics shipped in the package (min, max, mean, std of the training inputs)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from training.learning_package.examples import package_from_examples

SETS = [("iris", load_iris), ("wine", load_wine), ("digits", load_digits), ("breast_cancer", load_breast_cancer)]


def stats(x):
    x = np.asarray(x); return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=np.maximum(x.std(0), 1e-6))


def oor_frac(s, x, margin=0.1):
    rng = np.maximum(s["max"] - s["min"], 1e-6)
    return ((x < s["min"] - margin * rng) | (x > s["max"] + margin * rng)).mean(1)


def maxz(s, x, floor=0.0):
    sd = np.maximum(s["std"], floor * np.maximum(s["max"] - s["min"], 1e-6))   # floor: near-constant features must not make z explode
    return (np.abs(x - s["mean"]) / sd).max(1)


RULES = {
    "R0 any feature outside range+-10% (current)": lambda s, x: oor_frac(s, x) > 0,
    "R1 >=25% of features outside range+-10%": lambda s, x: oor_frac(s, x) >= 0.25,
    "R2 any feature |z|>8": lambda s, x: maxz(s, x) > 8,
    "R3 any feature |z|>12": lambda s, x: maxz(s, x) > 12,
    "R4 >=25% outside range+-10% OR any |z|>12": lambda s, x: (oor_frac(s, x) >= 0.25) | (maxz(s, x) > 12),
    "R5 >=10% outside range+-10% OR any |z|>12": lambda s, x: (oor_frac(s, x) >= 0.10) | (maxz(s, x) > 12),
    "R6 >=2 features outside range+-10% OR any |z|>12": lambda s, x: (oor_frac(s, x) * s["min"].size >= 2 - 1e-9) | (maxz(s, x) > 12),
    "R7 >=2 features outside range+-10% OR any |z|>8": lambda s, x: (oor_frac(s, x) * s["min"].size >= 2 - 1e-9) | (maxz(s, x) > 8),
    "R8 >=2 outside range+-10% OR any |z|>12 (std floored at 5% of range)": lambda s, x: (oor_frac(s, x) * s["min"].size >= 2 - 1e-9) | (maxz(s, x, 0.05) > 12),
    "R9 >=2 outside range+-10% OR any |z|>8 (std floored at 5% of range)": lambda s, x: (oor_frac(s, x) * s["min"].size >= 2 - 1e-9) | (maxz(s, x, 0.05) > 8),
}


def shifts(s, x, rng):
    sd = s["std"]
    yield "gross (x*10+1000)", x * 10 + 1000
    yield "moderate (+4 sd on every feature)", x + 4 * sd
    yield "moderate negative (-4 sd on every feature)", x - 4 * sd
    one = x.copy(); idx = rng.integers(0, x.shape[1], len(x)); one[np.arange(len(x)), idx] = s["mean"][idx] + 25 * sd[idx]
    yield "single feature +25 sd", one
    yield "unit-box uniform (in range, undetectable by range rules)", rng.uniform(s["min"], s["max"], x.shape)


def main():
    rng = np.random.default_rng(0); out = {}
    for name, loader in SETS:
        d = loader(); lp, (xte, yte), _ = package_from_examples(name, "t", ["k"], [str(i) for i in range(len(set(d.target)))], d.data.tolist(), d.target.tolist(), seed=0)
        s = stats(lp.train_x); xte = np.asarray(xte)
        res = {}
        for rname, rule in RULES.items():
            res[rname] = {"false_refusal_on_heldout": float(rule(s, xte).mean())}
            for sname, xs in shifts(s, xte, rng):
                res[rname][sname] = float(rule(s, xs).mean())
        out[name] = res
    # aggregate: mean over datasets
    agg = {r: {k: float(np.mean([out[n][r][k] for n in out])) for k in next(iter(out.values()))[r]} for r in RULES}
    return {"per_dataset": out, "mean_over_datasets": agg}


if __name__ == "__main__":
    r = main()
    for rname, v in r["mean_over_datasets"].items():
        print(rname); print("   " + " | ".join(f"{k.split(' (')[0]}: {x:.2f}" for k, x in v.items()))
    Path(Path(__file__).resolve().parent.parent / "benchmarks/reports/novelty_rules.json").write_text(json.dumps(r, indent=1))
