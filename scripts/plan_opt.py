"""Route optimisation: common-subexpression and dead-node elimination for straight-line (SSA) plans.
Capability calls are treated as pure functions of their inputs. Plans with loops, branches or re-assigned slots are left untouched."""
from __future__ import annotations
import copy, json


def _slots(refs):
    return [r["slot"] for r in refs if "slot" in r]


def _reads(n):
    keys = ("args", "options", "pair")
    out = []
    for k in keys:
        out += _slots(n.get(k, []))
    for k in ("index", "flag", "cond"):
        if k in n:
            out += _slots([n[k]])
    return out


def _writes(n):
    return [n["out"], n["out"] + ".conf"] if n["op"] == "cap" else [n["out"]]


def _rename(n, mapping):
    def fix(r):
        if "slot" in r and r["slot"] in mapping:
            return {**r, "slot": mapping[r["slot"]]}
        return r
    for k in ("args", "options", "pair"):
        if k in n:
            n[k] = [fix(r) for r in n[k]]
    for k in ("index", "flag", "cond"):
        if k in n:
            n[k] = fix(n[k])


def optimise(plan: dict) -> tuple[dict, dict]:
    p = copy.deepcopy(plan)
    stats = {"nodes_before": len(p["nodes"]), "cap_calls_before": sum(n["op"] == "cap" for n in p["nodes"])}
    if any(n["op"] in ("if", "repeat") for n in p["nodes"]):
        stats.update(applicable=False, nodes_after=stats["nodes_before"], cap_calls_after=stats["cap_calls_before"]); return p, stats
    written = [w for n in p["nodes"] for w in _writes(n)]
    if len(written) != len(set(written)):
        stats.update(applicable=False, nodes_after=stats["nodes_before"], cap_calls_after=stats["cap_calls_before"]); return p, stats
    seen, mapping, nodes = {}, {}, []
    for n in p["nodes"]:
        _rename(n, mapping)
        if n["op"] == "cap":
            key = json.dumps({k: n.get(k) for k in ("capability", "intent", "args")}, sort_keys=True)
            if key in seen:
                mapping[n["out"]] = seen[key]; mapping[n["out"] + ".conf"] = seen[key] + ".conf"; continue
            seen[key] = n["out"]
        nodes.append(n)
    for r in p["outputs"]:
        if "slot" in r and r["slot"] in mapping:
            r["slot"] = mapping[r["slot"]]
    # dead-node elimination (iterate to a fixpoint)
    changed = True
    while changed:
        used = set(_slots(p["outputs"]))
        for n in nodes:
            used |= set(_reads(n))
        keep = [n for n in nodes if any(w in used for w in _writes(n))]
        changed = len(keep) != len(nodes); nodes = keep
    p["nodes"] = nodes
    stats.update(applicable=True, nodes_after=len(nodes), cap_calls_after=sum(n["op"] == "cap" for n in nodes))
    return p, stats
