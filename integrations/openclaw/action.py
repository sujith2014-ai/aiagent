"""External-action interface. Everything outside the core (web, files, shell, other AIs) goes through an ActionProvider,
and every call is gated by the broker's policy check. Evidence is *data*, never instructions and never ground truth."""
from __future__ import annotations
import hashlib, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


class ActionDenied(Exception):
    def __init__(self, decision: dict):
        super().__init__(f"denied by policy rule '{decision.get('rule_id')}': {decision.get('reason')}")
        self.decision = decision


class NeedsApproval(Exception):
    def __init__(self, decision: dict):
        super().__init__(f"operator approval required: {decision.get('reason')}")
        self.decision = decision


class ActionUnavailable(Exception):
    """The provider could not perform the action (no provider configured, network blocked, timeout...). Never retried silently."""
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message = kind, message


@dataclass
class Evidence:
    id: str
    provider: str            # e.g. "openclaw:duckduckgo" or "fixture-simulated"
    source: str              # url or query that produced it
    excerpt: str             # truncated text, treated as untrusted data
    retrieved_at: str
    sha256: str              # hash of the full retrieved text
    simulated: bool = False
    trust: str = "unverified"

    def public(self) -> dict:
        return asdict(self)


def make_evidence(i: int, provider: str, source: str, text: str, simulated: bool = False, excerpt_len: int = 600) -> Evidence:
    return Evidence(id=f"ev{i}", provider=provider, source=source, excerpt=text[:excerpt_len],
                    retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sha256=hashlib.sha256(text.encode()).hexdigest(), simulated=simulated)


class ActionProvider(ABC):
    name = "abstract"

    @abstractmethod
    def search(self, query: str, limit: int = 3) -> list[Evidence]: ...

    @abstractmethod
    def fetch(self, url: str) -> list[Evidence]: ...
