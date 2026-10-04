"""TeacherProvider interface. Implementations (Qwen, OpenAI, Claude, local, ...) plug in here.
The teacher module deliberately has no access to signing code or keys."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from training.learning_package.schema import LearningPackage


@dataclass
class HelpRequest:
    """Minimum-required context for escalation (no history, no files, no secrets)."""
    task_intent: str
    input_dim: int
    known_capabilities: list[str] = field(default_factory=list)
    attempted_route: dict[str, Any] = field(default_factory=dict)
    failure: str = ""
    available_tools: list[str] = field(default_factory=list)


class TeacherProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def respond(self, req: HelpRequest) -> LearningPackage:
        """Return a LearningPackage (to be validated, never trusted) or raise TeacherCannotHelp."""


class TeacherCannotHelp(Exception):
    pass
