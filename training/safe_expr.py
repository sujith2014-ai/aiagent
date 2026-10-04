"""Restricted expression evaluator for teacher-supplied rules. Never uses eval()/exec(): the expression is parsed
with `ast` and walked by an explicit whitelist evaluator with node-count and magnitude limits."""
from __future__ import annotations
import ast, math, operator

MAX_NODES = 200
MAX_ABS = 1e9
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCS = {"abs": abs, "min": min, "max": max, "sum": sum, "round": round, "sqrt": math.sqrt, "int": int, "float": float, "len": len}


class UnsafeExpression(ValueError):
    pass


def compile_expr(expr: str, names: set[str]) -> ast.AST:
    if len(expr) > 500:
        raise UnsafeExpression("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(f"syntax error: {e}")
    n = 0
    for node in ast.walk(tree):
        n += 1
        if n > MAX_NODES:
            raise UnsafeExpression("expression too complex")
        if not isinstance(node, (ast.Expression, ast.IfExp, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
                                 ast.Load, ast.Subscript, ast.Call, ast.Tuple, ast.List, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Pow,
                                 *tuple(_BIN), *tuple(_CMP))):
            raise UnsafeExpression(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _FUNCS:
            raise UnsafeExpression(f"unknown name '{node.id}'")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords):
            raise UnsafeExpression("only whitelisted function calls allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise UnsafeExpression("only numeric constants allowed")
    return tree


def _chk(v):
    if isinstance(v, (int, float)) and abs(v) > MAX_ABS:
        raise UnsafeExpression("value too large")
    return v


def _ev(node, env):
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise UnsafeExpression(f"name '{node.id}' is not a value")
    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, ast.BoolOp):
        vals = (_ev(v, env) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, env)
        return (not v) if isinstance(node.op, ast.Not) else (-v if isinstance(node.op, ast.USub) else +v)
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left, env), _ev(node.right, env)
        if isinstance(node.op, ast.Pow):
            if not (isinstance(b, (int, float)) and 0 <= b <= 4 and abs(a) <= 1e6):
                raise UnsafeExpression("unsafe power")
            return _chk(a ** b)
        return _chk(_BIN[type(node.op)](a, b))
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, env)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        seq = _ev(node.value, env)
        idx = _ev(node.slice, env)
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise UnsafeExpression("index must be an integer")
        return seq[idx]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e, env) for e in node.elts]
    if isinstance(node, ast.Call):
        args = [_ev(a, env) for a in node.args]
        f = _FUNCS[node.func.id]
        if len(args) == 1 and isinstance(args[0], list):
            return _chk(f(args[0]))
        return _chk(f(*args))
    raise UnsafeExpression(f"cannot evaluate {type(node).__name__}")


def evaluate(tree: ast.AST, x: list[float]):
    env = {"x": list(x)}
    env.update({f"x{i}": v for i, v in enumerate(x)})
    return _ev(tree, env)


def names_for(dim: int) -> set[str]:
    return {"x"} | {f"x{i}" for i in range(dim)}
