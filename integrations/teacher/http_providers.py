"""Real teacher providers over HTTPS. They only ever send a `HelpRequest` (minimum context). API keys are read from
environment variables at call time and never logged or placed in prompts.
Validated against a local mock server (tests/test_phase6.py); live validation against real endpoints is PENDING
(needs network egress + API key: see docs/RUNBOOKS.md)."""
from __future__ import annotations
import json, os, urllib.request, urllib.error
from integrations.teacher.provider import TeacherProvider, HelpRequest

SYSTEM_PROMPT = """You are the teacher of a small modular AI that lacks a capability. Reply with ONE JSON object and nothing else.
Fields: "action" in [new_capability_spec, reroute, use_memory, external_research, request_tool, cannot_help], "rationale" (string).
For new_capability_spec include "spec": {"capability_id","description","keywords"[],"labels"[],"input_dim","domain":[{"lo","hi","grid"?}] per input,
"label_expr" (a single Python-style expression over x0,x1,... or x[i], using only + - * / // % ** comparisons, and/or/not, if-else, abs min max sum round sqrt;
it must evaluate to an integer label index), "worked_examples":[{"x":[...],"y":int}] (at least 3)}.
For reroute include "reroute_intent" and "capability_id" of an installed capability. For use_memory include "memory_text".
For external_research include "research_query". For request_tool include "tool". Never include code, URLs to execute, or credentials."""


class TeacherUnavailable(Exception):
    pass


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        raise TeacherUnavailable(f"{type(e).__name__}: {e}")


class OpenAICompatProvider(TeacherProvider):
    """Any OpenAI-compatible chat-completions endpoint (Qwen via DashScope/vLLM/Ollama, OpenAI, local servers)."""
    def __init__(self, base_url: str, model: str, api_key_env: str | None = "OPENAI_API_KEY", timeout: float = 60.0, name: str | None = None):
        self.base_url, self.model, self.api_key_env, self.timeout = base_url.rstrip("/"), model, api_key_env, timeout
        self.name = name or f"openai-compat:{model}"

    def advise(self, req: HelpRequest) -> str:
        headers = {}
        key = os.environ.get(self.api_key_env or "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": req.to_json()}]}
        out = _post(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise TeacherUnavailable("unexpected response shape")


class AnthropicProvider(TeacherProvider):
    def __init__(self, model: str, base_url: str = "https://api.anthropic.com", api_key_env: str = "ANTHROPIC_API_KEY", timeout: float = 60.0):
        self.model, self.base_url, self.api_key_env, self.timeout = model, base_url.rstrip("/"), api_key_env, timeout
        self.name = f"anthropic:{model}"

    def advise(self, req: HelpRequest) -> str:
        key = os.environ.get(self.api_key_env, "")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"model": self.model, "max_tokens": 2000, "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": req.to_json()}]}
        out = _post(f"{self.base_url}/v1/messages", headers, body, self.timeout)
        try:
            return "".join(b.get("text", "") for b in out["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise TeacherUnavailable("unexpected response shape")
