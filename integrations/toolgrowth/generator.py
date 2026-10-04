"""Tool generators. The interface is what a Teacher/LLM would implement; only a scripted simulator exists here (no model was called, see docs/RUNBOOKS.md R5)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from integrations.toolgrowth.package import Candidate


@dataclass
class ToolRequest:
    task_id: str
    intent: str
    description: str
    visible_examples: list      # a few input/expected pairs; hidden spec cases are never part of a request


class GeneratorUnavailable(Exception):
    pass


class ToolGenerator(ABC):
    name = "abstract"

    @abstractmethod
    def generate(self, request: ToolRequest, attempt: int, feedback: dict | None) -> Candidate | None:
        """Return a candidate, or None to give up. `feedback` is the previous attempt's failed stage and generic message only."""


class ScriptedGenerator(ToolGenerator):
    name = "scripted"

    def __init__(self, plan: dict):
        self.plan, self.seen_feedback, self.calls = plan, [], 0

    def generate(self, request, attempt, feedback):
        self.calls += 1; self.seen_feedback.append(feedback)
        seq = self.plan.get(request.task_id, [])
        return seq[min(attempt, len(seq) - 1)] if seq else None


class LLMGenerator(ToolGenerator):
    name = "llm"

    def generate(self, request, attempt, feedback):
        raise GeneratorUnavailable("no live model configured: docs/RUNBOOKS.md R5")
