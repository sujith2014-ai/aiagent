"""Run generated tool code in a child process: rlimits (CPU, address space, file size, fds), new network namespace, empty environment, empty working directory, restricted builtins and imports,
and a PEP 578 audit hook that kills the process on file/network/process/native access. HONEST LIMIT: this is a research-grade sandbox for CPython on Linux, not a production boundary against a determined
attacker (CPython audit hooks and builtin restriction have known bypass classes; a production deployment needs seccomp/gVisor/WASM). The layers can be switched off individually so experiments can measure each."""
from __future__ import annotations
import json, os, resource, shutil, signal, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from integrations.toolgrowth.static import ALLOWED_IMPORTS

RUNNER = Path(__file__).with_name("runner.py")


@dataclass
class SandboxConfig:
    timeout_s: float = 5.0
    cpu_s: int = 4
    mem_mb: int = 512
    out_bytes: int = 1 << 20
    netns: bool = True
    hook: bool = True
    builtins: bool = True


@dataclass
class SandboxResult:
    ok: bool
    results: list = field(default_factory=list)       # per input: {"ok": bool, "output"|"error": ...}
    kind: str = "ok"                                   # ok | timeout | violation | limit | crash | bad_output | fatal
    detail: str = ""
    seconds: float = 0.0
    network_isolated: bool = False


def netns_available() -> bool:
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


_NETNS = None


def run_batch(source: str, inputs: list, cfg: SandboxConfig | None = None) -> SandboxResult:
    global _NETNS
    cfg = cfg or SandboxConfig()
    if _NETNS is None: _NETNS = netns_available()
    use_netns = cfg.netns and _NETNS
    work = Path(tempfile.mkdtemp(prefix="toolbox-"))
    try:
        (work / "tool.py").write_text(source); (work / "in.json").write_text(json.dumps(inputs)); (work / "allowed.json").write_text(json.dumps(sorted(ALLOWED_IMPORTS)))
        flags = ",".join(f for f, on in (("builtins", cfg.builtins), ("hook", cfg.hook)) if on) or "none"
        cmd = (["unshare", "-rn", "--"] if use_netns else []) + [sys.executable, "-I", str(RUNNER), str(work), flags]

        def limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_s, cfg.cpu_s)); resource.setrlimit(resource.RLIMIT_AS, (cfg.mem_mb << 20, cfg.mem_mb << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (cfg.out_bytes, cfg.out_bytes)); resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64)); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        t0 = time.perf_counter()
        with open(work / "stdout", "wb") as so, open(work / "stderr", "wb") as se:
            p = subprocess.Popen(cmd, cwd=work, env={"PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL, stdout=so, stderr=se, preexec_fn=limits)
            try:
                rc = p.wait(timeout=cfg.timeout_s); timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                rc = p.wait()
        secs = time.perf_counter() - t0
        err = (work / "stderr").read_bytes()[:4000].decode(errors="replace")
        base = dict(seconds=secs, network_isolated=use_netns)
        if timed_out: return SandboxResult(False, kind="timeout", detail=f"exceeded {cfg.timeout_s}s wall time", **base)
        if "VIOLATION:" in err: return SandboxResult(False, kind="violation", detail=err.split("VIOLATION:", 1)[1].splitlines()[0], **base)
        if rc in (-signal.SIGXCPU, -signal.SIGXFSZ): return SandboxResult(False, kind="limit", detail=f"killed by {signal.Signals(-rc).name}", **base)
        if rc == -signal.SIGKILL: return SandboxResult(False, kind="limit", detail="killed by SIGKILL (CPU or address-space limit)", **base)
        out = work / "out.json"
        if rc != 0 or not out.exists() or out.stat().st_size == 0:
            return SandboxResult(False, kind="crash", detail=f"exit {rc}: {err.strip()[-300:]}", **base)
        try: data = json.loads(out.read_text())
        except ValueError: return SandboxResult(False, kind="bad_output", detail="result is not valid JSON", **base)
        if isinstance(data, dict) and "fatal" in data: return SandboxResult(False, kind="fatal", detail=data["fatal"], **base)
        return SandboxResult(True, results=data, **base)
    finally:
        shutil.rmtree(work, ignore_errors=True)
