"""PyTorch -> ONNX export for the tiny MLP modules.
The .cap spec only declares `model.format`; nothing outside this file knows about PyTorch."""
from __future__ import annotations
import io, torch, onnx, numpy as np
import onnxruntime as ort


def export_onnx(model: torch.nn.Module, input_dim: int) -> bytes:
    model.eval()
    buf = io.BytesIO()
    torch.onnx.export(model, torch.zeros(1, input_dim), buf, input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamo=False)
    data = buf.getvalue()
    onnx.checker.check_model(onnx.load_from_string(data))
    return data


def verify_export(model: torch.nn.Module, onnx_bytes: bytes, xs, tol=1e-4) -> float:
    """Max abs difference between PyTorch and onnxruntime logits on xs."""
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    arr = np.asarray(xs, dtype=np.float32)[:200]
    with torch.no_grad():
        ref = model(torch.from_numpy(arr)).numpy()
    out = np.concatenate([sess.run(None, {"input": a[None, :]})[0] for a in arr])  # batch-1 graph
    diff = float(np.abs(ref - out).max())
    if diff > tol:
        raise AssertionError(f"ONNX export mismatch {diff}")
    return diff
