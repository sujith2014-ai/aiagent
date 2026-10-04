"""Shared harness for Phase 12 tests/experiments: drives the Rust CLI's on-device learning commands with a device-local signing key."""
import json, subprocess, shutil
from pathlib import Path
from scripts.cli import AICLI, Cli


class Device:
    def __init__(self, root: Path, trust: Path, key_id="device-1", device="PC_CONSTRAINED", prefix=None, with_key=True):
        self.root, self.trust, self.device, self.prefix = Path(root), Path(trust), device, prefix or []
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_file = self.root.parent / f"{self.root.name}.devkey.json"
        self.key_id = key_id
        self.pub_b64 = None
        if with_key:
            out = subprocess.run([str(AICLI), "--root", "/x", "--trust", "/x", "keygen", "--out", str(self.key_file), "--key-id", key_id], capture_output=True, text=True, check=True)
            self.pub_b64 = json.loads(out.stdout)["public_b64"]
        self.cli = Cli(self.root, self.trust, device, prefix)
        if with_key: self.cli.extra = ["--device-key", str(self.key_file)]

    def _cmd(self, *args, with_key=True):
        base = [str(AICLI), "--root", str(self.root), "--trust", str(self.trust), "--device", self.device] + (["--device-key", str(self.key_file)] if with_key and self.key_file.exists() else [])
        return subprocess.run(self.prefix + base + [str(a) for a in args], capture_output=True, text=True)

    def _json(self, *args, **kw):
        p = self._cmd(*args, **kw)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[:400])
        return json.loads(p.stdout)

    def learn(self, spec: dict, with_key=True):
        f = self.root.parent / f"{self.root.name}.learn.json"; f.write_text(json.dumps(spec)); return self._json("learn", "--spec", f, with_key=with_key)

    def adapt(self, spec: dict):
        f = self.root.parent / f"{self.root.name}.adapt.json"; f.write_text(json.dumps(spec)); return self._json("adapt", "--spec", f)

    def alias(self, cap: str, keywords: list):
        return self._json("alias", cap, "--keywords", ",".join(keywords))

    def active_package(self, cap: str) -> Path:
        e = next(c for c in self.cli.list() if c["capability_id"] == cap)
        return self.root / "store" / e["store_file"]
