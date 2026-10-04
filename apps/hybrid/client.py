"""Device-side hybrid client. Placement is a small deterministic function (privacy first, local first); delivery is verified end to end:
signed catalog (key in the device's trust roots, monotonic sequence, maximum age) -> size/fit check -> download pinned to the catalog hash ->
the core's full activation sequence (signature, hashes, compatibility, sandbox load, bundled tests, resource check). The device holds public keys only."""
from __future__ import annotations
import base64, hashlib, json, time, urllib.request, urllib.error
from dataclasses import dataclass
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

LOCAL, SERVER, QUEUE, REFUSE = "LOCAL", "SERVER", "QUEUE", "REFUSE_KEEP_LOCAL"


def placement(local_result: str, sensitive: bool, online: bool, allow_server: bool) -> tuple[str, str]:
    """Deterministic placement. Local first; sensitive data never leaves the device; nothing is pretended while offline."""
    if local_result == "ANSWER":
        return LOCAL, "answered by an installed capability on this device"
    if sensitive:
        return REFUSE, "task is marked sensitive: it is not sent anywhere"
    if not allow_server:
        return REFUSE, "server execution is not permitted for this task"
    if not online:
        return QUEUE, "server unreachable: queued, nothing was executed"
    return SERVER, "no local capability: executed on the server"


class DeliveryError(Exception):
    pass


@dataclass
class SyncReport:
    installed: list; skipped: list; rejected: list; warnings: list; proposed_revocations: list


class HybridClient:
    def __init__(self, local, server_url: str, token: str, workdir: Path, trust_path: Path, max_model_bytes: int, max_catalog_age_s: float = 7 * 86400, timeout: float = 20.0, allow_server_solve=True):
        self.local, self.url, self.token = local, server_url.rstrip("/"), token
        self.work = Path(workdir); self.work.mkdir(parents=True, exist_ok=True)
        self.trust_path, self.max_model_bytes, self.max_age, self.timeout, self.allow_server = Path(trust_path), max_model_bytes, max_catalog_age_s, timeout, allow_server_solve
        self.state_file = self.work / "sync_state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"last_sequence": 0, "proposed_revocations": []}
        self.queue_file = self.work / "pending_actions.jsonl"
        self.requests_to_server = 0

    # ---- transport --------------------------------------------------------------------------------------------
    def _http(self, method, path, body=None, raw=False, max_bytes=64 << 20):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                     headers={"Authorization": f"Bearer {self.token}", **({"Content-Type": "application/json"} if body is not None else {})})
        self.requests_to_server += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = r.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise DeliveryError("response exceeds size limit")
                return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            raise DeliveryError(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise ConnectionError(str(e))

    def online(self) -> bool:
        try:
            self.requests_to_server -= 0
            req = urllib.request.Request(self.url + "/health"); urllib.request.urlopen(req, timeout=3).read(); return True
        except Exception:
            return False

    # ---- signed catalog ---------------------------------------------------------------------------------------
    def _trust(self) -> dict:
        t = json.loads(self.trust_path.read_text())
        revoked = set(t.get("__revoked__", []))
        return {k: v for k, v in t.items() if k != "__revoked__" and k not in revoked}

    def verify_catalog(self, env: dict) -> dict:
        sig = env.get("signature", {}); body = env.get("body", "")
        pub = self._trust().get(sig.get("key_id"))
        if sig.get("alg") != "ed25519" or pub is None:
            raise DeliveryError(f"catalog signed by untrusted or revoked key '{sig.get('key_id')}'")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(pub)).verify(base64.b64decode(sig["sig_b64"]), body.encode())
        except (InvalidSignature, KeyError, ValueError):
            raise DeliveryError("catalog signature invalid")
        cat = json.loads(body)
        if cat["sequence"] < self.state["last_sequence"]:
            raise DeliveryError(f"catalog replay: sequence {cat['sequence']} < last seen {self.state['last_sequence']}")
        if time.time() - cat["generated_at"] > self.max_age:
            raise DeliveryError("catalog is older than the maximum allowed age (possible freeze attack)")
        return cat

    # ---- sync: server -> device ---------------------------------------------------------------------------------
    def sync(self) -> SyncReport:
        rep = SyncReport([], [], [], [], [])
        cat = self.verify_catalog(self._http("GET", "/catalog"))
        self.state["last_sequence"] = cat["sequence"]
        if cat.get("revoked_keys"):
            self.state["proposed_revocations"] = sorted(set(self.state["proposed_revocations"]) | set(cat["revoked_keys"]))   # proposed only: the operator applies them
            rep.proposed_revocations = self.state["proposed_revocations"]
        self.state_file.write_text(json.dumps(self.state))
        installed = {c["capability_id"]: c["active_version"] for c in self.local.list()}
        latest: dict[str, dict] = {}
        for p in cat["packages"]:
            if p["capability_id"] not in latest or tuple(map(int, p["version"].split("."))) > tuple(map(int, latest[p["capability_id"]]["version"].split("."))):
                latest[p["capability_id"]] = p
        for cap, p in sorted(latest.items()):
            have = installed.get(cap)
            if have is not None:
                if tuple(map(int, p["version"].split("."))) < tuple(map(int, have.split("."))):
                    rep.warnings.append(f"{cap}: catalog offers {p['version']} but {have} is installed (possible downgrade attempt); ignored"); continue
                if p["version"] == have:
                    rep.skipped.append((cap, "up to date")); continue
            if p["model_bytes"] > self.max_model_bytes:
                rep.skipped.append((cap, f"model of {p['model_bytes']} bytes does not fit this device (limit {self.max_model_bytes}); use server execution")); continue
            if not (p["package_id"].replace("-", "").replace("_", "").replace(".", "").isalnum()):
                rep.rejected.append((cap, "suspicious package id")); continue
            data = self._http("GET", f"/packages/{p['package_id']}.cap", raw=True, max_bytes=max(p["size"], 1) + 1024)
            if hashlib.sha256(data).hexdigest() != p["sha256"] or len(data) != p["size"]:
                rep.rejected.append((cap, "downloaded bytes do not match the signed catalog entry")); continue
            f = self.work / f"dl-{p['package_id']}.cap"; f.write_bytes(data)
            try:
                r = self.local.import_caps(str(f))[0]
            finally:
                f.unlink(missing_ok=True)
            (rep.installed if r["activated"] else rep.rejected).append((cap, p["version"] if r["activated"] else f"core rejected at {r['failed_step']}"))
        return rep

    def apply_revocations(self) -> list:
        """Operator action: add the proposed key ids to this device's trust roots. Never called automatically."""
        t = json.loads(self.trust_path.read_text()); cur = set(t.get("__revoked__", []))
        new = [k for k in self.state["proposed_revocations"] if k not in cur]
        t["__revoked__"] = sorted(cur | set(new)); self.trust_path.write_text(json.dumps(t))
        return new

    # ---- placement ----------------------------------------------------------------------------------------------
    def solve(self, intent: str, x: list[float], sensitive: bool = False) -> dict:
        local = self.local.solve(intent, x)
        decision, why = placement(local["result"], sensitive, self.online() if local["result"] != "ANSWER" and not sensitive and self.allow_server else False, self.allow_server)
        out = {"placement": decision, "why": why}
        if decision == LOCAL:
            out.update(local); out["executed_on"] = "device"
        elif decision == SERVER:
            try:
                r = self._http("POST", "/solve", {"intent": intent, "input": x}); out.update(r)
            except (ConnectionError, DeliveryError) as e:
                self._queue({"type": "solve", "intent": intent, "input": x}); out.update(placement=QUEUE, why=f"server call failed ({e}); queued", result="NEEDS_HELP")
        elif decision == QUEUE:
            self._queue({"type": "solve", "intent": intent, "input": x}); out["result"] = "NEEDS_HELP"
        else:
            out["result"] = "NEEDS_HELP"; out["local_reason"] = local.get("reason_code")
        return out

    # ---- teach (heavy training is offloaded; examples are uploaded only with explicit consent) -------------------------
    def teach(self, spec: dict, sensitive: bool = False, consent_to_upload: bool = False) -> dict:
        if sensitive or not consent_to_upload:
            return {"uploaded": False, "reason": "examples are sensitive or upload was not consented to; nothing left the device"}
        try:
            r = self._http("POST", "/train", spec)
        except (ConnectionError, DeliveryError) as e:
            self._queue({"type": "teach", "spec": spec}); return {"uploaded": False, "queued": True, "reason": f"server unreachable ({e}); queued locally"}
        if r.get("learned"):
            r["sync"] = self.sync()
        return {"uploaded": True, **r}

    # ---- offline queue -------------------------------------------------------------------------------------------------
    def _queue(self, entry: dict):
        with open(self.queue_file, "a") as f:
            f.write(json.dumps({"ts": time.time(), **entry}) + "\n")

    def pending(self) -> list:
        return [json.loads(l) for l in self.queue_file.read_text().splitlines() if l.strip()] if self.queue_file.exists() else []

    def flush(self) -> dict:
        items = self.pending()
        if not items:
            return {"flushed": 0}
        if not self.online():
            return {"flushed": 0, "still_queued": len(items), "reason": "server unreachable"}
        self.queue_file.unlink(); done = []
        for it in items:
            try:
                if it["type"] == "solve":
                    done.append(self._http("POST", "/solve", {"intent": it["intent"], "input": it["input"]}))
                else:
                    done.append(self.teach(it["spec"], consent_to_upload=True))
            except (ConnectionError, DeliveryError):
                self._queue(it)
        return {"flushed": len(done) , "results": done, "still_queued": len(self.pending())}
