"""Tool generator reached through an AI chat website (OpenClaw browser), no model API key. It sends only the ToolRequest (never the hidden spec cases) and, on retries, the pipeline's generic feedback.
The reply is untrusted: it is parsed into a Candidate and goes through every gate (static analysis, sandbox, tests, independent spec, operator approval) like any other candidate."""
from __future__ import annotations
import json
from integrations.teacher.browser_teacher import BrowserChat, extract_json
from integrations.teacher.http_providers import TeacherUnavailable
from integrations.toolgrowth.generator import ToolGenerator, ToolRequest, GeneratorUnavailable
from integrations.toolgrowth.package import Candidate
from integrations.toolgrowth.static import ALLOWED_IMPORTS

CONTRACT = """You write a small pure Python tool. Reply with ONE JSON object and nothing else: {"description": str, "keywords": [at least 2 distinct lowercase words], "source": str, "tests": [{"input": {...}, "expected": {...}}]}.
The source must define `def run(inp): ...` taking one dict and returning one dict; it must be deterministic and stateless. Allowed imports only: """ + ", ".join(sorted(ALLOWED_IMPORTS)) + """.
No files, network, processes, eval/exec/open/getattr, classes, globals, str.format, or names starting with an underscore. Include at least 2 tests that match the examples."""


class BrowserToolGenerator(ToolGenerator):
    name = "browser"

    def __init__(self, chat: BrowserChat):
        self.chat = chat; self.name = f"browser:{chat.site.name}"; self.seen_prompts: list[str] = []

    def generate(self, request: ToolRequest, attempt: int, feedback: dict | None) -> Candidate | None:
        body = {"task": request.description, "intent": request.intent, "examples": request.visible_examples, "attempt": attempt}
        if feedback: body["previous_attempt_failed"] = feedback
        prompt = " ".join(CONTRACT.split()) + " TOOL_REQUEST: " + json.dumps(body)
        self.seen_prompts.append(prompt)
        try: d = json.loads(extract_json(self.chat.ask(prompt)))
        except TeacherUnavailable as e: raise GeneratorUnavailable(str(e))
        try:
            return Candidate(request.task_id, f"0.1.{attempt}" if attempt else "0.1.0", str(d["description"])[:200], str(d["source"]), tests=list(d["tests"]), keywords=[str(k).lower() for k in d.get("keywords", [])],
                             permissions=list(d.get("permissions", [])), generator=self.name)
        except (KeyError, TypeError, ValueError):
            return None                                    # malformed reply: counts as a rejected attempt
