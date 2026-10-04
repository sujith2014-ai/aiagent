"""A tool candidate: pure-function Python source plus a manifest the system can hash and the operator can approve.
A tool is `def run(inp: dict) -> dict`. In v1 a tool is granted NO permissions: no files, network, processes or native code."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, field

ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
VER_RE = re.compile(r"^\d{1,4}\.\d{1,4}\.\d{1,4}$")
MAX_SOURCE_BYTES = 20_000
MAX_TESTS = 200


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def request_hash_of(params: dict) -> str:
    """Mirrors core/src/policy.rs::request_hash (serde_json sorts keys of a BTreeMap)."""
    return hashlib.sha256(json.dumps({"action": "tool.install", "params": params}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass
class Candidate:
    tool_id: str
    version: str
    description: str
    source: str
    tests: list = field(default_factory=list)            # [{"input": {...}, "expected": {...}}] written by the generator
    permissions: list = field(default_factory=list)      # what the generator DECLARES it needs; v1 grants none
    keywords: list = field(default_factory=list)
    generator: str = "unknown"

    def tests_text(self) -> str:
        return "\n".join(json.dumps(t, sort_keys=True) for t in self.tests)

    def code_sha256(self) -> str:
        return sha(self.source)

    def tests_sha256(self) -> str:
        return sha(self.tests_text())

    def approval_params(self) -> dict:
        """Exactly what the operator approves. Any change to code, tests, version or permissions changes the request hash and voids the approval."""
        return {"tool_id": self.tool_id, "version": self.version, "code_sha256": self.code_sha256(), "tests_sha256": self.tests_sha256(), "permissions": sorted(self.permissions), "keywords": sorted(self.keywords)}

    def manifest(self) -> dict:
        return {"format": "tool/1", "tool_id": self.tool_id, "version": self.version, "description": self.description, "keywords": sorted(self.keywords),
                "entry": "run", "language": "python3", "permissions": sorted(self.permissions), "code_sha256": self.code_sha256(), "tests_sha256": self.tests_sha256(), "generator": self.generator}

    def schema_problems(self) -> list[str]:
        p = []
        if not ID_RE.match(self.tool_id or ""): p.append("tool_id must match ^[a-z][a-z0-9_]{2,40}$")
        if not VER_RE.match(self.version or ""): p.append("version must be MAJOR.MINOR.PATCH")
        if not self.description or len(self.description) > 200: p.append("description is required and at most 200 characters")
        if len(self.source.encode()) > MAX_SOURCE_BYTES: p.append(f"source larger than {MAX_SOURCE_BYTES} bytes")
        if not (1 <= len(self.tests) <= MAX_TESTS): p.append(f"between 1 and {MAX_TESTS} bundled tests are required")
        if not all(isinstance(t, dict) and isinstance(t.get("input"), dict) and "expected" in t for t in self.tests): p.append("each test needs an object 'input' and an 'expected'")
        if self.permissions: p.append(f"permissions {sorted(self.permissions)} requested, but this runtime grants no permissions to generated tools")
        return p
