"""Thin Python wrapper around the Rust `aicli` binary."""
from __future__ import annotations
import json, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AICLI = ROOT / "target" / "release" / "aicli"


class Cli:
    def __init__(self, root: Path, trust: Path, device="PC_FULL", prefix=None):
        self.root, self.trust, self.device, self.prefix = Path(root), Path(trust), device, prefix or []

    def _run(self, *args, check=True):
        cmd = self.prefix + [str(AICLI), "--root", str(self.root), "--trust", str(self.trust), "--device", self.device, *map(str, args)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if check and p.returncode != 0:
            raise RuntimeError(f"aicli failed: {p.stderr}")
        return p

    def json(self, *args):
        return json.loads(self._run(*args).stdout)

    def import_caps(self, *paths):
        return self.json("import", *paths)

    def solve(self, intent, x):
        return self.json("solve", "--intent", intent, "--input", ",".join(repr(float(v)) for v in x))

    def batch(self, cases_file):
        return self.json("batch", "--cases", cases_file)

    def list(self):
        return self.json("list")

    def regression(self, cap):
        return self.json("regression", cap)
