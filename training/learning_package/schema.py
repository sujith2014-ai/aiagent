"""LearningPackage v1: the only thing trainers accept from a teacher.
Teacher output is never trained on directly; it is parsed into this structure and validated first."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json

SCHEMA_VERSION = "learning-package/1"
STRATEGIES = {"no_change", "memory_update", "knowledge_update", "router_update", "adapter",
              "adapt_existing", "new_connection", "module_expansion", "new_module"}
INFO_KINDS = {"MEMORY", "KNOWLEDGE", "CAPABILITY"}


@dataclass
class LearningPackage:
    info_kind: str                      # MEMORY | KNOWLEDGE | CAPABILITY
    strategy: str                       # one of STRATEGIES
    capability_id: str
    description: str
    keywords: list[str]
    input_dim: int
    labels: list[str]
    device_caps: list[str] = field(default_factory=list)
    train_x: list[list[float]] = field(default_factory=list)
    train_y: list[int] = field(default_factory=list)
    edge_cases_x: list[list[float]] = field(default_factory=list)
    edge_cases_y: list[int] = field(default_factory=list)
    counterexamples: list[dict[str, Any]] = field(default_factory=list)  # {"x": [...], "not_label": int}
    validation_x: list[list[float]] = field(default_factory=list)
    validation_y: list[int] = field(default_factory=list)
    existing_dependencies: list[str] = field(default_factory=list)
    suggested_modification: str = ""
    regression_capabilities: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "LearningPackage":
        return LearningPackage(**json.loads(s))


class InvalidLearningPackage(ValueError):
    pass


def validate(lp: LearningPackage, known_capabilities: set[str] | None = None) -> None:
    """Raise InvalidLearningPackage on any structural or logical defect."""
    err = []
    if lp.schema_version != SCHEMA_VERSION:
        err.append(f"schema_version {lp.schema_version}")
    if lp.info_kind not in INFO_KINDS:
        err.append(f"info_kind {lp.info_kind}")
    if lp.strategy not in STRATEGIES:
        err.append(f"strategy {lp.strategy}")
    if lp.info_kind != "CAPABILITY" and lp.strategy in {"new_module", "adapter", "adapt_existing", "module_expansion", "new_connection"}:
        err.append(f"info_kind {lp.info_kind} must not trigger neural training (strategy {lp.strategy})")
    if lp.info_kind == "CAPABILITY" and lp.strategy == "new_module":
        n = len(lp.labels)
        if n < 2:
            err.append("need >= 2 labels")
        if not lp.capability_id or not lp.keywords:
            err.append("capability_id and keywords required")
        if len(lp.train_x) != len(lp.train_y) or len(lp.train_x) < 50:
            err.append("train examples missing/mismatched (need >= 50)")
        if len(lp.validation_x) != len(lp.validation_y) or len(lp.validation_x) < 20:
            err.append("validation tests missing/mismatched (need >= 20)")
        for xs, ys, nm in [(lp.train_x, lp.train_y, "train"), (lp.validation_x, lp.validation_y, "validation"),
                           (lp.edge_cases_x, lp.edge_cases_y, "edge")]:
            if any(len(x) != lp.input_dim for x in xs):
                err.append(f"{nm}: example with wrong dimension")
            if any(not (0 <= y < n) for y in ys):
                err.append(f"{nm}: label out of range")
        if len(set(map(tuple, lp.train_x)) & set(map(tuple, lp.validation_x))) > 0:
            err.append("validation leaks into train")
        for c in lp.counterexamples:
            if len(c.get("x", [])) != lp.input_dim or not (0 <= c.get("not_label", -1) < n):
                err.append("bad counterexample")
        if known_capabilities is not None:
            for d in lp.existing_dependencies:
                if d not in known_capabilities:
                    err.append(f"unknown dependency {d}")
    if not lp.provenance.get("source"):
        err.append("provenance.source required")
    if err:
        raise InvalidLearningPackage("; ".join(err))
