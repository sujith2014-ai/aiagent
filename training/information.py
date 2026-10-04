"""Information classification before any neural training: MEMORY / KNOWLEDGE / CAPABILITY.
Rule-based baseline (Phase 5). Only CAPABILITY may lead to neural training."""
from __future__ import annotations
import re

_CAP = re.compile(r"\b(learn(?:ing)? (?:to|how to)|be able to|get better at|classify|recogni[sz]e|detect|predict|compare|sort|rank|estimate|identify|distinguish)\b", re.I)
_MEM = re.compile(r"\b(my|i|we|our|me)\b", re.I)
_MEM_FACT = re.compile(r"\b(is|are|was|were|am|parked|left|put|live[sd]?|keep|kept|located|on level|at)\b", re.I)
_KNOW = re.compile(r"\b(api|apis|library|tool|uses|use|requires|supports|endpoint|oauth|version|protocol|format|documentation|default|returns)\b", re.I)


def classify_information(text: str) -> str:
    t = text.strip()
    if _CAP.search(t) and not re.match(r"^(my|i|we|our)\b", t, re.I):
        return "CAPABILITY"
    if _MEM.search(t) and _MEM_FACT.search(t) and not _KNOW.search(t):
        return "MEMORY"
    if _KNOW.search(t):
        return "KNOWLEDGE"
    return "KNOWLEDGE"
