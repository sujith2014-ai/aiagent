"""SIMULATED action provider for validating the research loop without network egress. Evidence it returns is marked
`simulated=True` and provider `fixture-simulated`, so reports can never confuse it with live OpenClaw output."""
from __future__ import annotations
from integrations.openclaw.action import ActionProvider, ActionUnavailable, make_evidence


class FixtureProvider(ActionProvider):
    name = "fixture-simulated"

    def __init__(self, search_results: dict[str, list[tuple[str, str]]] | None = None, pages: dict[str, str] | None = None, fail: str | None = None):
        self.search_results, self.pages, self.fail = search_results or {}, pages or {}, fail
        self.calls: list[tuple[str, str]] = []

    def search(self, query, limit=3):
        self.calls.append(("search", query))
        if self.fail:
            raise ActionUnavailable(self.fail, "simulated failure")
        hits = next((v for k, v in self.search_results.items() if k in query.lower()), [])
        if not hits:
            raise ActionUnavailable("no_results", "simulated: no results")
        return [make_evidence(i, self.name, url, text, simulated=True) for i, (url, text) in enumerate(hits[:limit])]

    def fetch(self, url):
        self.calls.append(("fetch", url))
        if self.fail:
            raise ActionUnavailable(self.fail, "simulated failure")
        if url not in self.pages:
            raise ActionUnavailable("no_results", "simulated: unknown page")
        return [make_evidence(0, self.name, url, self.pages[url], simulated=True)]
