"""TeacherProvider interface. Real providers (Qwen, OpenAI, Claude, Gemini, local) implement `advise` and return the
raw text/JSON the model produced; it is parsed and validated by `response.parse_response` and never trusted.
This module deliberately has no access to key-handling or package-building code."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import re
from typing import Any


@dataclass
class HelpRequest:
    """Minimum-required context for escalation: no history, no files, no secrets, no raw input values."""
    task_intent: str
    input_dim: int
    known_capabilities: list[str] = field(default_factory=list)
    attempted_route: dict[str, Any] = field(default_factory=dict)
    failure: str = ""
    available_tools: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        import json
        return json.dumps(asdict(self))


_SECRET_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)\b\s*[:=]\s*\S+"), "[SECRET]"),
    (re.compile(r"\b(sk|pk|ghp|xox[a-z])[-_][A-Za-z0-9_-]{8,}\b"), "[SECRET]"),
    (re.compile(r"\b[A-Fa-f0-9]{24,}\b"), "[TOKEN]"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{32,}={0,2}"), "[TOKEN]"),
    (re.compile(r"(?:[A-Za-z]:\\|/)(?:[\w .-]+[\\/])+[\w .-]*"), "[PATH]"),
    (re.compile(r"\b\d{12,19}\b"), "[NUMBER]"),
]


def redact(text: str, max_len: int = 200) -> str:
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text[:max_len]


def make_help_request(intent: str, input_dim: int, known: list[str], decision: dict | None = None, failure: str = "",
                      tools: list[str] | None = None) -> HelpRequest:
    d = decision or {}
    route = {"status": d.get("status") or d.get("reason_code"), "candidates": [c.get("capability_id") for c in d.get("candidates", [])][:3]}
    return HelpRequest(redact(intent), int(input_dim), sorted(known), route, redact(failure, 120), list(tools or []))


class TeacherProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def advise(self, req: HelpRequest) -> str:
        """Return the teacher's raw JSON answer for the request (validated downstream)."""


class TeacherCannotHelp(Exception):
    pass
