"""Teacher simulator: stands in for a real LLM teacher in Phase 1-5.
It owns hidden deterministic generators ("ground truth") and turns a help request into a
LearningPackage. The student's modules never see the generator code, only examples."""
from __future__ import annotations
import random, time
from typing import Callable
from integrations.teacher.provider import TeacherProvider, HelpRequest, TeacherCannotHelp
from training.learning_package.schema import LearningPackage

GRID = 20  # numeric comparison uses a discrete grid so EQUAL occurs


def _gen_compare(rng):          # two numbers -> LESS / EQUAL / GREATER
    a, b = rng.randrange(GRID), rng.randrange(GRID)
    return [a / (GRID - 1), b / (GRID - 1)], (0 if a < b else 1 if a == b else 2)

def _gen_region(rng):           # point in plane -> INSIDE / OUTSIDE circle r^2 = 0.5
    x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
    return [x, y], 0 if x * x + y * y < 0.5 else 1

def _gen_argmax(rng):           # which of 4 values is largest
    v = [rng.uniform(0, 1) for _ in range(4)]
    return v, max(range(4), key=lambda i: v[i])

def _gen_majority(rng):         # 5 binary-ish values: is the majority > 0.5 ?  (used for tool/other tests)
    v = [rng.random() for _ in range(5)]
    return v, int(sum(x > 0.5 for x in v) >= 3)


TASKS: dict[str, dict] = {
    "compare_numbers": dict(gen=_gen_compare, dim=2, labels=["LESS", "EQUAL", "GREATER"], dedupe=True, n_train=240, n_val=80, n_test=80,
        description="Compare two numbers and classify their relation",
        keywords=["compare", "comparison", "numbers", "number", "relation", "less", "greater", "equal", "larger", "smaller", "ordering"],
        match=["compare", "comparison", "relation", "ordering"]),
    "point_region": dict(gen=_gen_region, dim=2, labels=["INSIDE", "OUTSIDE"], dedupe=False, n_train=1500, n_val=300, n_test=1000,
        description="Decide whether a 2D point lies inside the central circular region",
        keywords=["point", "region", "inside", "outside", "circle", "circular", "geometry", "plane", "coordinates"],
        match=["region", "inside", "circle", "geometry"]),
    "argmax_position": dict(gen=_gen_argmax, dim=4, labels=["P0", "P1", "P2", "P3"], dedupe=False, n_train=1500, n_val=300, n_test=1000,
        description="Find the position of the largest of four values",
        keywords=["largest", "maximum", "max", "argmax", "position", "index", "values", "biggest", "highest"],
        match=["largest", "maximum", "argmax", "biggest"]),
}


def sample(task: str, n: int, seed: int, exclude: set | None = None, unique=False):
    t = TASKS[task]
    rng = random.Random(seed)
    xs, ys, seen = [], [], set(exclude or ())
    tries = 0
    while len(xs) < n and tries < n * 50:
        tries += 1
        x, y = t["gen"](rng)
        key = tuple(x)
        if (unique or t["dedupe"]) and key in seen:
            continue
        seen.add(key)
        xs.append(x); ys.append(y)
    return xs, ys


def splits(task: str, seed: int = 1234):
    """Canonical disjoint splits: train / val / test(bundled) / eval(independent where the domain allows).
    compare_numbers has only 400 distinct pairs, so eval == test there (documented limitation)."""
    t = TASKS[task]
    trx, try_ = sample(task, t["n_train"], seed)
    used = set(map(tuple, trx))
    vx, vy = sample(task, t["n_val"], seed + 1, exclude=used, unique=True)
    used |= set(map(tuple, vx))
    tx, ty = sample(task, t["n_test"], seed + 100, exclude=used, unique=True)
    if t["dedupe"]:
        ex, ey = tx, ty
    else:
        ex, ey = sample(task, t["n_test"], seed + 200, exclude=used | set(map(tuple, tx)), unique=True)
    return dict(train=(trx, try_), val=(vx, vy), test=(tx, ty), eval=(ex, ey))


class TeacherSimulator(TeacherProvider):
    name = "teacher-simulator/1"

    def __init__(self, seed=1234):
        self.seed = seed
        self.calls = 0

    def heldout(self, task: str, exclude: set, seed_offset=100):
        """Independent unseen examples for the build/validation service (oracle use, Phase 1 only)."""
        t = TASKS[task]
        return sample(task, t["n_test"], self.seed + seed_offset, exclude=exclude, unique=True)

    def _identify(self, intent: str, dim: int) -> str:
        toks = set(intent.lower().replace(",", " ").split())
        for name, t in TASKS.items():
            if t["dim"] == dim and toks & set(t["match"]):
                return name
        raise TeacherCannotHelp(f"simulator knows no task for '{intent}' (dim {dim})")

    def respond(self, req: HelpRequest) -> LearningPackage:
        self.calls += 1
        name = self._identify(req.task_intent, req.input_dim)
        t = TASKS[name]
        trx, try_ = sample(name, t["n_train"], self.seed)
        used = set(map(tuple, trx))
        vx, vy = sample(name, t["n_val"], self.seed + 1, exclude=used, unique=True)
        # edge cases: boundaries / ties the teacher knows are tricky
        ex, ey = [], []
        if name == "compare_numbers":
            for v in (0, 5, 19):
                ex.append([v / 19, v / 19]); ey.append(1)
            ex.append([0.0, 1.0]); ey.append(0); ex.append([1.0, 0.0]); ey.append(2)
        elif name == "argmax_position":
            ex.append([0.5, 0.5001, 0.5, 0.5]); ey.append(1)
        counter = [{"x": [0.0, 1.0], "not_label": 2}] if name == "compare_numbers" else []
        return LearningPackage(
            info_kind="CAPABILITY", strategy="new_module", capability_id=name,
            description=t["description"], keywords=t["keywords"], input_dim=t["dim"], labels=t["labels"],
            train_x=trx, train_y=try_, edge_cases_x=ex, edge_cases_y=ey, counterexamples=counter,
            validation_x=vx, validation_y=vy,
            suggested_modification="train a new small module; do not modify existing modules",
            provenance={"source": "teacher-simulator", "teacher": self.name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "verified": True, "verification": "synthetic ground-truth generator", "request_intent": req.task_intent},
        )
