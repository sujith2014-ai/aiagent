"""Build + sign `.cap` packages. Lives in the *build/validation service* side; the teacher side never imports this."""
from __future__ import annotations
import base64, hashlib, io, json, time, zipfile
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

FORMAT_VERSION = "cap/1"
RUNTIME_MIN = "0.1.0"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def keygen(key_id: str, keydir: Path) -> None:
    keydir.mkdir(parents=True, exist_ok=True)
    k = Ed25519PrivateKey.generate()
    (keydir / f"{key_id}.private").write_bytes(k.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))
    (keydir / f"{key_id}.private").chmod(0o600)
    pub = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    (keydir / f"{key_id}.public").write_text(base64.b64encode(pub).decode())


def load_private(key_id: str, keydir: Path) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes((keydir / f"{key_id}.private").read_bytes())


def write_trust(path: Path, keys: dict[str, str]) -> None:
    path.write_text(json.dumps(keys, indent=2))


def build_cap(out: Path, *, capability_id: str, version: str, model_bytes: bytes, params: int, input_dim: int,
              labels: list[str], tests: list[tuple[list[float], int]], keywords: list[str], description: str,
              signer_id: str, signer_key: Ed25519PrivateKey, provenance: dict, min_accuracy: float,
              device_caps: list[str] | None = None, dependencies: list[str] | None = None,
              model_format: str = "onnx", variant: str = "fp32", runtime_min: str = RUNTIME_MIN,
              min_ram_mb: int = 64, format_version: str = FORMAT_VERSION,
              input_stats: dict | None = None, calibration: dict | None = None) -> dict:
    tests_blob = ("\n".join(json.dumps({"input": x, "expected": y}) for x, y in tests) + "\n").encode()
    hints_blob = json.dumps({"description": description, "keywords": keywords}, sort_keys=True).encode()
    model_name = f"model/model.{model_format}"
    files = {model_name: model_bytes, "tests/cases.jsonl": tests_blob, "routing/hints.json": hints_blob}
    pkg_id = f"{capability_id}-{version}-{sha256(model_bytes)[:8]}"
    manifest = {
        "format_version": format_version, "package_id": pkg_id, "capability_id": capability_id, "version": version,
        "model": {"format": model_format, "variant": variant, "file": model_name},
        "io": {"input": {"dtype": "f32", "dim": input_dim, "description": "feature vector"},
               "output": {"kind": "classification", "labels": labels}},
        "requires": {"runtime_min": runtime_min, "device_caps": device_caps or []},
        "dependencies": dependencies or [],
        "resources": {"params": params, "model_bytes": len(model_bytes), "min_ram_mb": min_ram_mb},
        "provenance": provenance,
        "tests": {"file": "tests/cases.jsonl", "count": len(tests), "min_accuracy": min_accuracy},
        "routing_file": "routing/hints.json",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signer": {"key_id": signer_id},
        **({"input_stats": input_stats} if input_stats else {}),
        **({"calibration": calibration} if calibration else {}),
        "contents": {k: sha256(v) for k, v in sorted(files.items())},
    }
    mbytes = json.dumps(manifest, sort_keys=True, indent=1).encode()
    sig = {"alg": "ed25519", "key_id": signer_id, "sig_b64": base64.b64encode(signer_key.sign(mbytes)).decode()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", mbytes)
        z.writestr("signature.json", json.dumps(sig))
        for k, v in files.items():
            z.writestr(k, v)
    return manifest
