"""Fixtures for the Kotlin/JVM bridge tests: signed packages (the *same* .cap files the desktop uses), hostile variants, and expected
outputs computed independently by the Rust CLI, so the JNI path can be checked against them."""
import json, random, shutil, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import Cli, ROOT
from scripts import plans as P
from packages.capbuild import keygen, write_trust, build_cap, load_private
from integrations.teacher.simulator import TeacherSimulator, splits
from integrations.teacher.provider import HelpRequest
from training.service import BuildService


def main(out: Path):
    if out.exists():
        shutil.rmtree(out)
    (out / "caps").mkdir(parents=True); (out / "bad").mkdir()
    keygen("build-svc-1", out / "keys"); keygen("rogue", out / "keys")
    write_trust(out / "trust.json", {"build-svc-1": (out / "keys/build-svc-1.public").read_text()})
    T = TeacherSimulator(); svc = BuildService(out / "keys", "build-svc-1", out / "build", T)
    ref = Cli(out / "ref-runtime", out / "trust.json")
    intents = {"compare_numbers": ("compare numbers relation", 2), "point_region": ("point inside region", 2), "argmax_position": ("largest value position", 4)}
    canon = {"compare_numbers": "compare these two numbers", "point_region": "is this point inside the circular region", "argmax_position": "find the position of the largest value"}
    known = set()
    for cap, (it, dim) in intents.items():
        b = svc.build(T.respond(HelpRequest(it, dim)), "0.1.0", known); assert b.promoted
        shutil.copy(b.cap_path, out / "caps" / f"{cap}.cap"); assert ref.import_caps(str(b.cap_path))[0]["activated"]; known.add(cap)
    # expected outputs, computed by the Rust CLI on the reference runtime
    rng = random.Random(7); solves = []
    for cap in intents:
        ex = splits(cap)["eval"]
        for i in rng.sample(range(len(ex[0])), 20):
            r = ref.solve(canon[cap], ex[0][i]); assert r["result"] == "ANSWER"
            solves.append({"capability": cap, "intent": canon[cap], "input": ex[0][i], "label": r["label"], "label_index": r["label_index"], "probs": r["probs"], "status": r["status"]})
    ood = ref.solve(canon["compare_numbers"], [5.0, 3.0]); unknown = ref.solve("translate this sentence to french", [0.1, 0.2])
    plan = P.sort4_network("intent"); v = [0.9, 0.2, 0.7, 0.4]
    pf = out / "plan.json"; pf.write_text(json.dumps(plan)); cf = out / "plan_cases.jsonl"; cf.write_text(json.dumps({"inputs": {"v": v}}) + "\n")
    plan_out = ref.plan_batch(str(pf), str(cf))["results"][0]["outputs"]
    # hostile / incompatible packages
    good = out / "caps" / "compare_numbers.cap"
    def repack(dst, replace=None):
        with zipfile.ZipFile(good) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for n in zi.namelist():
                zo.writestr(n, (replace or {}).get(n, zi.read(n)))
    data = bytearray(zipfile.ZipFile(good).read("model/model.onnx")); data[len(data) // 2] ^= 0xFF
    repack(out / "bad/tampered.cap", {"model/model.onnx": bytes(data)})
    m = svc.models["compare_numbers"]
    common = dict(capability_id="compare_numbers", version="0.9.0", model_bytes=m["onnx"], params=m["params"], input_dim=2, labels=m["labels"], tests=m["tests"],
                  keywords=["compare"], description="d", provenance={"source": "fixture"}, min_accuracy=0.9, input_stats=m["stats"])
    build_cap(out / "bad/untrusted.cap", signer_id="rogue", signer_key=load_private("rogue", out / "keys"), **common)
    build_cap(out / "bad/needs_camera.cap", signer_id="build-svc-1", signer_key=svc.key, device_caps=["camera"], **{**common, "capability_id": "camera_thing"})
    (out / "bad/garbage.cap").write_bytes(b"not a zip")
    (out / "expectations.json").write_text(json.dumps({"solves": solves, "ood": {"result": ood["result"], "reason_code": ood.get("reason_code")}, "unknown": {"result": unknown["result"]},
                                                       "plan": {"inputs": {"v": v}, "outputs": plan_out}}))
    shutil.copy(ROOT / "integrations/openclaw/policy.default.json", out / "policy.json")
    print("fixtures in", out, "solves:", len(solves))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
