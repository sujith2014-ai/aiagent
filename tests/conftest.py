import sys, json, shutil, zipfile
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cli import Cli
from packages.capbuild import keygen, write_trust, load_private, build_cap
from integrations.teacher.simulator import TeacherSimulator, splits
from integrations.teacher.provider import HelpRequest
from training.service import BuildService
from training.trainer.train import train_candidate
from training.exporters.onnx_export import export_onnx


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("env")
    keygen("build-svc-1", d / "keys"); keygen("rogue", d / "keys")
    trust = d / "trust.json"
    write_trust(trust, {"build-svc-1": (d / "keys" / "build-svc-1.public").read_text()})
    teacher = TeacherSimulator()
    svc = BuildService(d / "keys", "build-svc-1", d / "build", teacher)
    lp = teacher.respond(HelpRequest("compare numbers relation", 2))
    good = svc.build(lp, "0.1.0", set())
    assert good.promoted
    model, _ = train_candidate(lp, seed=5)
    sp = splits("compare_numbers")
    return dict(dir=d, trust=trust, good=good.cap_path, lp=lp, onnx=export_onnx(model, 2), tests=list(zip(*sp["test"])),
                keyfn=lambda k: load_private(k, d / "keys"), teacher=teacher, svc=svc)


@pytest.fixture
def cli(env, tmp_path):
    return Cli(tmp_path / "rt", env["trust"])


def repack(src: Path, dst: Path, replace=None, add=None, drop=()):
    replace, add = replace or {}, add or {}
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
        for n in zi.namelist():
            if n in drop:
                continue
            zo.writestr(n, replace.get(n, zi.read(n)))
        for n, b in add.items():
            zo.writestr(n, b)
    return dst


def make_cap(env, out, key="build-svc-1", **over):
    kw = dict(capability_id="compare_numbers", version="0.1.0", model_bytes=env["onnx"], params=1251, input_dim=2,
              labels=["LESS", "EQUAL", "GREATER"], tests=env["tests"], keywords=env["lp"].keywords,
              description="d", signer_id=key, signer_key=env["keyfn"](key), provenance={"source": "test"}, min_accuracy=0.9)
    kw.update(over)
    return build_cap(out, **kw)
