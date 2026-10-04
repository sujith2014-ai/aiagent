"""Static analysis of generated tool source. Defence in depth, NOT the security boundary (the sandbox is): it rejects early, explains why, and derives the permissions the code would need,
so a tool that declares none but uses some is caught as an attempt to raise its own permissions."""
from __future__ import annotations
import ast
from dataclasses import dataclass

ALLOWED_IMPORTS = {"math", "json", "re", "statistics", "itertools", "collections", "datetime", "string", "decimal", "fractions", "textwrap", "unicodedata", "functools", "operator",
                   "heapq", "bisect", "numbers", "typing", "calendar", "csv"}
PERMISSION_OF_MODULE = {"socket": "net", "ssl": "net", "urllib": "net", "http": "net", "ftplib": "net", "smtplib": "net", "requests": "net", "asyncio": "net", "xmlrpc": "net",
                        "os": "fs", "pathlib": "fs", "shutil": "fs", "tempfile": "fs", "glob": "fs", "io": "fs", "fileinput": "fs", "sqlite3": "fs", "pickle": "fs", "shelve": "fs",
                        "subprocess": "exec", "multiprocessing": "exec", "threading": "exec", "pty": "exec", "signal": "exec", "sys": "exec", "importlib": "exec", "builtins": "exec",
                        "ctypes": "native", "cffi": "native", "mmap": "native", "marshal": "native", "gc": "native", "inspect": "native", "code": "native"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "help", "exit", "quit", "memoryview"}
PERMISSION_OF_CALL = {"open": "fs", "eval": "exec", "exec": "exec", "compile": "exec", "__import__": "exec", "getattr": "exec", "setattr": "exec", "globals": "exec", "locals": "exec", "vars": "exec"}
MAX_NODES = 6000


@dataclass
class Finding:
    rule: str
    line: int
    message: str

    def public(self) -> dict:
        return {"rule": self.rule, "line": self.line, "message": self.message}


def analyze(source: str, declared_permissions=()) -> dict:
    findings: list[Finding] = []
    derived: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"ok": False, "findings": [Finding("syntax", e.lineno or 0, str(e)).public()], "derived_permissions": []}
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES: findings.append(Finding("size", 0, f"{len(nodes)} AST nodes exceeds {MAX_NODES}"))

    def add(rule, node, msg): findings.append(Finding(rule, getattr(node, "lineno", 0), msg))

    has_run = any(isinstance(n, ast.FunctionDef) and n.name == "run" and len(n.args.args) == 1 for n in tree.body)
    if not has_run: findings.append(Finding("entry", 0, "a top-level `def run(inp)` taking exactly one argument is required"))
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    add("import", node, f"module '{a.name}' is not allowed"); 
                    if top in PERMISSION_OF_MODULE: derived.add(PERMISSION_OF_MODULE[top])
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or top not in ALLOWED_IMPORTS:
                add("import", node, f"import from '{'.' * node.level}{node.module}' is not allowed")
                if top in PERMISSION_OF_MODULE: derived.add(PERMISSION_OF_MODULE[top])
            if any(a.name == "*" for a in node.names): add("import", node, "star imports are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)): add("scope", node, "global/nonlocal are not allowed (tools must be stateless)")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)): add("construct", node, f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and node.attr != "__name__":   # __name__ is a read-only string and `type(x).__name__` is an everyday idiom (it was a false positive in the first run)
                add("attribute", node, f"access to private/dunder attribute '{node.attr}' is not allowed")
            if node.attr in ("format", "format_map"): add("attribute", node, "str.format can traverse attributes; use f-strings or concatenation")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"): add("name", node, f"name '{node.id}' is not allowed")
            if node.id in FORBIDDEN_CALLS and isinstance(getattr(node, "ctx", None), ast.Load):
                add("builtin", node, f"'{node.id}' is not allowed")
                if node.id in PERMISSION_OF_CALL: derived.add(PERMISSION_OF_CALL[node.id])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            add("string", node, "string constants containing '__' are not allowed (dynamic attribute/import tricks)")
    declared = set(declared_permissions)
    undeclared = sorted(derived - declared)
    if undeclared: findings.append(Finding("permission", 0, f"code needs permission(s) {undeclared} that the manifest does not declare"))
    return {"ok": not findings, "findings": [f.public() for f in findings], "derived_permissions": sorted(derived)}
