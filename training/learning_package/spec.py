"""Declarative task specs (what a real LLM teacher can reliably return) -> LearningPackage.
The teacher supplies a restricted rule + worked examples; labels are produced by the sandboxed evaluator."""
from __future__ import annotations
import itertools, random, time
from dataclasses import dataclass, field
from typing import Any
from training.learning_package.schema import LearningPackage
from training.safe_expr import compile_expr, evaluate, names_for, UnsafeExpression


class InvalidSpec(ValueError):
    pass


@dataclass
class TaskSpec:
    capability_id: str
    description: str
    keywords: list[str]
    labels: list[str]
    input_dim: int
    domain: list[dict[str, Any]]            # per input dim: {"lo": float, "hi": float, "grid": int (optional)}
    label_expr: str                         # restricted expression -> label index
    worked_examples: list[dict[str, Any]] = field(default_factory=list)   # [{"x": [...], "y": int}]
    n_train: int = 1500
    n_val: int = 300

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        try:
            s = TaskSpec(**{k: d[k] for k in ("capability_id", "description", "keywords", "labels", "input_dim", "domain", "label_expr")},
                         worked_examples=d.get("worked_examples", []), n_train=int(d.get("n_train", 1500)), n_val=int(d.get("n_val", 300)))
        except Exception as e:
            raise InvalidSpec(f"malformed spec: {e}")
        if not (1 <= s.input_dim <= 16) or len(s.domain) != s.input_dim or len(s.labels) < 2 or len(s.labels) > 32:
            raise InvalidSpec("bad dimensions/labels")
        if not (50 <= s.n_train <= 5000 and 20 <= s.n_val <= 2000):
            raise InvalidSpec("sample sizes out of bounds")
        for dm in s.domain:
            if not (isinstance(dm.get("lo"), (int, float)) and isinstance(dm.get("hi"), (int, float)) and dm["lo"] < dm["hi"]):
                raise InvalidSpec("bad domain")
            if "grid" in dm and not (2 <= int(dm["grid"]) <= 1000):
                raise InvalidSpec("bad grid")
        return s


def _label(tree, x, n_labels):
    try:
        y = evaluate(tree, x)
    except (UnsafeExpression, ZeroDivisionError, IndexError, TypeError, ValueError, OverflowError) as e:
        raise InvalidSpec(f"rule failed on {x}: {e}")
    if isinstance(y, bool):
        y = int(y)
    if not isinstance(y, (int, float)) or int(y) != y or not (0 <= int(y) < n_labels):
        raise InvalidSpec(f"rule returned invalid label {y!r} on {x}")
    return int(y)


def sample_spec(spec: TaskSpec, n: int, seed: int, exclude: set | None = None, unique=False):
    tree = compile_expr(spec.label_expr, names_for(spec.input_dim))
    rng = random.Random(seed)
    xs, ys, seen = [], [], set(exclude or ())
    for _ in range(n * 50):
        if len(xs) >= n:
            break
        x = []
        for dm in spec.domain:
            if "grid" in dm:
                k = rng.randrange(int(dm["grid"])); x.append(dm["lo"] + k * (dm["hi"] - dm["lo"]) / (int(dm["grid"]) - 1))
            else:
                x.append(rng.uniform(dm["lo"], dm["hi"]))
        key = tuple(x)
        grid_only = all("grid" in dm for dm in spec.domain)
        if (unique or grid_only) and key in seen:
            continue
        seen.add(key); xs.append(x); ys.append(_label(tree, x, len(spec.labels)))
    if len(xs) < n:
        raise InvalidSpec(f"could only sample {len(xs)} of {n} distinct examples")
    return xs, ys


def check_worked_examples(spec: TaskSpec) -> int:
    tree = compile_expr(spec.label_expr, names_for(spec.input_dim))
    if len(spec.worked_examples) < 3:
        raise InvalidSpec("need at least 3 worked examples")
    for ex in spec.worked_examples:
        if _label(tree, ex["x"], len(spec.labels)) != ex["y"]:
            raise InvalidSpec(f"rule contradicts its own worked example {ex}")
    return len(spec.worked_examples)


def spec_to_learning_package(spec: TaskSpec, teacher_name: str, seed: int = 11) -> LearningPackage:
    n_ok = check_worked_examples(spec)
    grid_only = all("grid" in dm for dm in spec.domain)
    domain_size = 1
    if grid_only:
        for dm in spec.domain:
            domain_size *= int(dm["grid"])
    n_train, n_val = spec.n_train, spec.n_val
    if grid_only and n_train + n_val > domain_size * 0.7:   # finite domain: leave room for held-out examples
        n_train, n_val = int(domain_size * 0.5), int(domain_size * 0.2)
    trx, try_ = sample_spec(spec, n_train, seed)
    vx, vy = sample_spec(spec, n_val, seed + 1, exclude=set(map(tuple, trx)), unique=True)
    corners = [list(c) for c in itertools.islice(itertools.product(*[(dm["lo"], dm["hi"]) for dm in spec.domain]), 16)]
    tree = compile_expr(spec.label_expr, names_for(spec.input_dim))
    ex_x = [list(c) for c in corners] + [e["x"] for e in spec.worked_examples]
    ex_y = [_label(tree, x, len(spec.labels)) for x in ex_x]
    return LearningPackage(
        info_kind="CAPABILITY", strategy="new_module", capability_id=spec.capability_id, description=spec.description,
        keywords=spec.keywords, input_dim=spec.input_dim, labels=spec.labels, train_x=trx, train_y=try_,
        edge_cases_x=ex_x, edge_cases_y=ex_y, validation_x=vx, validation_y=vy,
        suggested_modification="train a new small module from the sandbox-labelled data",
        provenance={"source": f"teacher-spec:{teacher_name}", "teacher": teacher_name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "verified": False, "verification": f"rule consistent with {n_ok} worked examples; NOT independently verified until environment examples pass",
                    "label_expr": spec.label_expr})
