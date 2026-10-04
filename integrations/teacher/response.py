"""Structured teacher output + strict validation. Anything unparsable or inconsistent is rejected."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any
from training.learning_package.schema import LearningPackage

ACTIONS = {"new_capability_spec", "reroute", "use_memory", "external_research", "adapt_existing", "new_capability", "request_tool", "cannot_help"}


class InvalidTeacherResponse(ValueError):
    pass


@dataclass
class TeacherResponse:
    action: str
    rationale: str
    learning_package: LearningPackage | None = None
    reroute_intent: str | None = None
    capability_id: str | None = None
    research_query: str | None = None
    tool: str | None = None
    memory_text: str | None = None
    spec: dict | None = None


def parse_response(raw: str | dict, known_capabilities: set[str]) -> TeacherResponse:
    try:
        d: Any = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        raise InvalidTeacherResponse(f"not JSON: {e}")
    if not isinstance(d, dict):
        raise InvalidTeacherResponse("response is not an object")
    action = d.get("action")
    if action not in ACTIONS:
        raise InvalidTeacherResponse(f"unknown action {action!r}")
    r = TeacherResponse(action=action, rationale=str(d.get("rationale", ""))[:500])
    if action == "new_capability":
        try:
            r.learning_package = LearningPackage(**d["learning_package"])
        except Exception as e:
            raise InvalidTeacherResponse(f"bad learning_package: {e}")
    elif action == "new_capability_spec":
        if not isinstance(d.get("spec"), dict):
            raise InvalidTeacherResponse("spec object required")
        r.spec = d["spec"]
    elif action == "reroute":
        r.reroute_intent = d.get("reroute_intent"); r.capability_id = d.get("capability_id")
        if r.capability_id not in known_capabilities or not r.reroute_intent:
            raise InvalidTeacherResponse("reroute must name an installed capability and a new intent")
    elif action == "adapt_existing":
        r.capability_id = d.get("capability_id")
        if r.capability_id not in known_capabilities:
            raise InvalidTeacherResponse("adapt_existing must name an installed capability")
    elif action == "external_research":
        r.research_query = d.get("research_query")
        if not r.research_query:
            raise InvalidTeacherResponse("research_query required")
    elif action == "request_tool":
        r.tool = d.get("tool")
        if not r.tool:
            raise InvalidTeacherResponse("tool required")
    elif action == "use_memory":
        r.memory_text = d.get("memory_text")
        if not r.memory_text:
            raise InvalidTeacherResponse("memory_text required")
    return r
