"""Hand-written (deterministic planner baseline) execution plans over the three learned capabilities.
Capabilities are referenced by *intent* (resolved dynamically by the router) or by id."""
from __future__ import annotations

CMP, REG, ARG = "compare numbers relation", "point inside region", "position of largest value"


def S(slot, idx=None):
    return {"slot": slot} if idx is None else {"slot": slot, "index": idx}


def C(x):
    return {"const": x}


def cap(id_, target, args, out, by="intent"):
    n = {"op": "cap", "id": id_, "args": args, "out": out}
    n["intent" if by == "intent" else "capability"] = target
    return n


def cmp_target(by):
    return CMP if by == "intent" else "compare_numbers"


def reg_target(by):
    return REG if by == "intent" else "point_region"


def arg_target(by):
    return ARG if by == "intent" else "argmax_position"


def sort4_network(by="intent"):
    nodes = [{"op": "gather", "id": "w0", "args": [S("v")], "out": "w0"}]
    cur = "w0"
    for s, (i, j) in enumerate([(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]):
        nxt = f"w{s+1}"
        nodes += [cap(f"f{s}", cmp_target(by), [S(cur, i), S(cur, j)], f"f{s}", by),
                  {"op": "cond_swap", "id": f"p{s}", "pair": [S(cur, i), S(cur, j)], "flag": S(f"f{s}"), "swap_if": [2], "out": f"p{s}"}]
        args = [S(cur, k) for k in range(4)]
        args[i], args[j] = S(f"p{s}", 0), S(f"p{s}", 1)
        nodes.append({"op": "gather", "id": f"g{s}", "args": args, "out": nxt})
        cur = nxt
    return {"id": f"sort4_network_{by}", "inputs": {"v": 4}, "nodes": nodes, "outputs": [S(cur)]}


def sort4_bubble(by="intent"):
    body = []
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 3)]):
        args = [S("w", m) for m in range(4)]
        args[i], args[j] = S(f"p{k}", 0), S(f"p{k}", 1)
        body += [cap(f"f{k}", cmp_target(by), [S("w", i), S("w", j)], f"f{k}", by),
                 {"op": "cond_swap", "id": f"p{k}", "pair": [S("w", i), S("w", j)], "flag": S(f"f{k}"), "swap_if": [2], "out": f"p{k}"},
                 {"op": "gather", "id": f"g{k}", "args": args, "out": "w"}]
    return {"id": f"sort4_bubble_{by}", "inputs": {"v": 4},
            "nodes": [{"op": "gather", "id": "w0", "args": [S("v")], "out": "w"}, {"op": "repeat", "id": "passes", "times": 3, "body": body}],
            "outputs": [S("w")]}


def count_inside(by="intent"):
    nodes = [cap(f"b{k}", reg_target(by), [S("p", 2 * k), S("p", 2 * k + 1)], f"c{k}", by) for k in range(4)]
    nodes.append({"op": "affine", "id": "count", "args": [S(f"c{k}") for k in range(4)], "scale": -1.0, "offset": 4.0, "out": "n"})
    return {"id": f"count_inside_{by}", "inputs": {"p": 8}, "nodes": nodes, "outputs": [S("n")]}


def mixed(by="intent"):
    rot = [S("w", 1), S("w", 2), S("w", 3), S("w", 0)]
    nodes = [cap("c", arg_target(by), [S("v")], "i", by),
             {"op": "select", "id": "m", "index": S("i"), "options": [S("v", k) for k in range(4)], "out": "m"},
             {"op": "select", "id": "x", "index": S("i"), "options": [S("w", k) for k in range(4)], "out": "x"},
             {"op": "select", "id": "y", "index": S("i"), "options": rot, "out": "y"},
             cap("b", reg_target(by), [S("x"), S("y")], "r", by),
             cap("a", cmp_target(by), [S("m"), C(10 / 19)], "rel", by)]
    return {"id": f"mixed_{by}", "inputs": {"v": 4, "w": 4}, "nodes": nodes, "outputs": [S("r"), S("rel")]}


def branch(by="intent"):
    then = [cap("tb", reg_target(by), [S("p")], "o", by)]
    other = [cap("ec", arg_target(by), [S("v")], "o", by)]
    return {"id": f"branch_{by}", "inputs": {"a": 2, "p": 2, "v": 4},
            "nodes": [cap("ca", cmp_target(by), [S("a")], "r", by),
                      {"op": "if", "id": "br", "cond": S("r"), "in_set": [2], "then": then, "else": other}],
            "outputs": [S("r"), S("o")]}


def path_abac(by="intent"):
    return {"id": f"path_ABAC_{by}", "inputs": {"a": 2, "a2": 2, "p": 2, "v": 4},
            "nodes": [cap("A1", cmp_target(by), [S("a")], "o1", by), cap("B", reg_target(by), [S("p")], "o2", by),
                      cap("A2", cmp_target(by), [S("a2")], "o3", by), cap("C", arg_target(by), [S("v")], "o4", by)],
            "outputs": [S("o1"), S("o2"), S("o3"), S("o4")]}
