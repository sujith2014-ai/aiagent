"""Teacher simulator: stands in for a real LLM teacher in Phase 1-5.
It owns hidden deterministic generators ("ground truth") and turns a help request into a
LearningPackage. The student's modules never see the generator code, only examples."""
from __future__ import annotations
import random, time
from typing import Callable
import json
from dataclasses import asdict
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


def _gen_majority5(rng):        # 5 values: do most of them exceed one half?
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
    "majority_vote": dict(gen=_gen_majority5, dim=5, labels=["MINORITY", "MAJORITY"], dedupe=False, n_train=4000, n_val=600, n_test=1000,
        description="Decide whether most of five values exceed one half",
        keywords=["majority", "vote", "most", "values", "exceed", "half", "threshold", "above", "count"],
        match=["majority", "vote", "most"]),
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

    def __init__(self, seed=1234, mode="lp", fault=None):
        """mode: "lp" returns full LearningPackages; "spec" returns declarative specs like a real LLM would.
        fault (spec mode only): None | "contradicts_examples" | "plausible_wrong" | "malicious_expr" """
        self.seed = seed
        self.calls = 0
        self.mode, self.fault = mode, fault

    def heldout(self, task: str, exclude: set, seed_offset=100):
        """Independent unseen examples for the build/validation service (oracle use, Phase 1 only)."""
        t = TASKS[task]
        return sample(task, t["n_test"], self.seed + seed_offset, exclude=exclude, unique=True)

    # words a language model would associate with each task (stands in for semantic understanding)
    SEMANTIC = {"compare_numbers": {"order", "bigger", "smaller", "greater", "larger", "same", "equal", "less", "relation", "rank", "ordering"},
                "point_region": {"coordinate", "disc", "within", "inside", "outside", "circle", "circular", "region", "geometry"},
                "argmax_position": {"index", "slot", "winning", "biggest", "highest", "maximum", "largest", "entry", "max", "argmax"},
                "majority_vote": {"majority", "vote", "most", "exceed", "half", "above"}}
    CANON_INTENT = {"compare_numbers": "compare numbers relation", "point_region": "point inside region", "argmax_position": "largest value position", "majority_vote": "majority vote values"}

    def _identify(self, intent: str, dim: int) -> str:
        toks = set(intent.lower().replace(",", " ").split())
        for name, t in TASKS.items():
            if t["dim"] == dim and toks & (set(t["match"]) | self.SEMANTIC[name]):
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


    # ---- structured advice (what a real LLM teacher would return as JSON) ----
    MEMORY_MARKERS = {"remember", "my", "where", "parked", "reminder"}
    RESEARCH_MARKERS = {"weather", "latest", "news", "price", "today", "current", "forecast"}
    TOOL_MARKERS = {"send", "email", "open", "download", "delete", "install", "browse", "search"}

    def advise(self, req: HelpRequest) -> str:
        toks = set(req.task_intent.lower().replace(",", " ").split())
        research = self._research_flow(req)
        if research is not None:
            return research
        try:
            known_name = self._identify(req.task_intent, req.input_dim)
        except TeacherCannotHelp:
            known_name = None
        if known_name in req.known_capabilities:
            self.calls += 1
            return json.dumps({"action": "reroute", "rationale": "an installed capability already solves this", "reroute_intent": self.CANON_INTENT[known_name], "capability_id": known_name})
        if self.mode == "spec":
            try:
                name = self._identify(req.task_intent, req.input_dim)
                self.calls += 1
                return json.dumps({"action": "new_capability_spec", "rationale": "rule-based synthetic task", "spec": self.make_spec(name)})
            except TeacherCannotHelp:
                pass
        try:
            lp = self.respond(req)
            return json.dumps({"action": "new_capability", "rationale": "teachable synthetic capability", "learning_package": asdict(lp)})
        except TeacherCannotHelp:
            pass
        # tool and research requests are checked before the memory markers ("my" appears in many tool requests)
        if toks & self.TOOL_MARKERS:
            return json.dumps({"action": "request_tool", "rationale": "needs an external action", "tool": sorted(toks & self.TOOL_MARKERS)[0]})
        if toks & self.RESEARCH_MARKERS:
            return json.dumps({"action": "external_research", "rationale": "needs current external evidence", "research_query": req.task_intent})
        if toks & self.MEMORY_MARKERS:
            return json.dumps({"action": "use_memory", "rationale": "personal fact, not a skill", "memory_text": req.task_intent})
        return json.dumps({"action": "cannot_help", "rationale": "simulator has no knowledge of this task"})


    SPEC_EXPR = {
        "compare_numbers": "0 if x0 < x1 else 1 if x0 == x1 else 2",
        "point_region": "0 if x0*x0 + x1*x1 < 0.5 else 1",
        "argmax_position": "0 if x0 >= max(x1, x2, x3) else 1 if x1 >= max(x0, x2, x3) else 2 if x2 >= max(x0, x1, x3) else 3",
    }
    SPEC_DOMAIN = {"compare_numbers": [{"lo": 0.0, "hi": 1.0, "grid": GRID}] * 2, "point_region": [{"lo": -1.0, "hi": 1.0}] * 2,
                   "argmax_position": [{"lo": 0.0, "hi": 1.0}] * 4}

    def make_spec(self, name: str) -> dict:
        t = TASKS[name]
        rng = random.Random(self.seed + 5)
        ex = [t["gen"](rng) for _ in range(6)]
        expr = self.SPEC_EXPR[name]
        if self.fault == "contradicts_examples":
            expr = "2" if name == "compare_numbers" else "1" if name == "point_region" else "0"
        elif self.fault in ("plausible_wrong", "subtle_wrong"):
            # wrong on a region, but the worked examples are chosen (as a confident-but-mistaken teacher might) to agree with the rule
            gross = {"compare_numbers": "1 if (x0 > 0.5 and x1 > 0.5) else (0 if x0 < x1 else 1 if x0 == x1 else 2)",
                     "point_region": "0 if x0*x0 + x1*x1 < 0.5 and x0 < 0.3 else 1",
                     "argmax_position": "0 if x0 >= max(x1, x2, x3) else 1 if x1 >= max(x0, x2, x3) else 2 if x2 >= max(x0, x1, x3) and x3 < 0.8 else 3"}
            subtle = dict(gross, compare_numbers="1 if (x0 > 0.9 and x1 > 0.9) else (0 if x0 < x1 else 1 if x0 == x1 else 2)")
            expr = (gross if self.fault == "plausible_wrong" else subtle)[name]
            from training.safe_expr import compile_expr, evaluate, names_for
            tree = compile_expr(expr, names_for(t["dim"]))
            agree, rng2 = [], random.Random(self.seed + 6)
            while len(agree) < 6:
                x, y = t["gen"](rng2)
                if int(evaluate(tree, x)) == y:
                    agree.append((x, y))
            ex = agree
        elif self.fault == "malicious_expr":
            expr = "__import__('os').system('echo pwned')"
        return {"capability_id": name, "description": t["description"], "keywords": t["keywords"], "labels": t["labels"], "input_dim": t["dim"],
                "domain": self.SPEC_DOMAIN[name], "label_expr": expr, "worked_examples": [{"x": x, "y": y} for x, y in ex],
                "n_train": t["n_train"], "n_val": t["n_val"]}


    # ---- a capability whose rule exists only in external documents (research flow) ----
    RESEARCH_MARKERS_TASK = {"grade", "band", "bands"}

    def _research_flow(self, req: HelpRequest):
        """Stands in for an LLM that does not know a task's thresholds until it has read evidence. It "reads" evidence with a regex
        (plumbing/provenance test only: no claim about real language understanding) and ignores any instructions inside it."""
        toks = set(req.task_intent.lower().replace(",", " ").split())
        if req.input_dim != 1 or not (toks & self.RESEARCH_MARKERS_TASK):
            return None
        self.calls += 1
        if not req.evidence:
            return json.dumps({"action": "external_research", "rationale": "thresholds are defined in an external document",
                               "research_query": "grade band thresholds LOW MID HIGH score ranges"})
        import re
        for ev in req.evidence:
            m1 = re.search(r"below\s+([0-9]*\.?[0-9]+)\s+(?:are|is)\s+LOW", ev["excerpt"], re.I)
            m2 = re.search(r"up to\s+([0-9]*\.?[0-9]+)\s+(?:are|is)\s+MID", ev["excerpt"], re.I)
            if m1 and m2:
                t1, t2 = float(m1.group(1)), float(m2.group(1))
                spec = {"capability_id": "grade_band", "description": "Classify a score into LOW, MID or HIGH bands", "keywords": ["grade", "band", "score", "low", "mid", "high", "bands", "thresholds"],
                        "labels": ["LOW", "MID", "HIGH"], "input_dim": 1, "domain": [{"lo": 0.0, "hi": 1.0}], "label_expr": f"0 if x0 < {t1} else 1 if x0 < {t2} else 2",
                        "worked_examples": [{"x": [t1 / 2], "y": 0}, {"x": [(t1 + t2) / 2], "y": 1}, {"x": [min(1.0, t2 + (1 - t2) / 2)], "y": 2}],
                        "n_train": 1500, "n_val": 300, "evidence_ids": [ev["id"]]}
                return json.dumps({"action": "new_capability_spec", "rationale": f"thresholds read from evidence {ev['id']}", "spec": spec})
        return json.dumps({"action": "cannot_help", "rationale": "evidence did not contain parseable thresholds"})
