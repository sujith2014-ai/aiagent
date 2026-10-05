"""Phase 14 lab: a synthetic long-running world (36 capabilities, rule drift, paraphrases, unsupported requests), a world-aware teacher, a seeded arrival stream,
the modular system under test (escalator + build service + monitor + consolidation) and conventional single-network baselines."""
from __future__ import annotations
import json, random, time, collections
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from training.drift import Cusum, WindowMonitor, ProbeScheduler
from integrations.teacher.provider import TeacherProvider, HelpRequest, make_help_request
from integrations.teacher.response import parse_response
from integrations.escalation import Escalator
from training.learning_package.spec import TaskSpec, sample_spec

NOUNS = ["boiler", "pump", "valve", "rotor", "tank", "cable", "filter", "sensor", "fan", "gear", "motor", "relay", "piston", "bearing", "turbine", "heater", "cooler", "battery", "antenna",
         "mirror", "lens", "magnet", "spring", "brake", "clutch", "nozzle", "gauge", "fuse", "coil", "panel", "drill", "lathe", "press", "mixer", "burner", "chiller"]
FAMILIES = ["alarm", "zone", "side", "grade", "parity", "vote", "peak", "order"]
SYN = {"alarm": ["warning", "trigger"], "zone": ["region", "area"], "side": ["half", "orientation"], "grade": ["tier", "class"], "parity": ["mismatch", "agreement"],
       "vote": ["consensus", "poll"], "peak": ["maximum", "top"], "order": ["rank", "sequence"]}
LABELS = {"alarm": ["OK", "ALARM"], "zone": ["INSIDE", "OUTSIDE"], "side": ["LEFT", "RIGHT"], "grade": ["LOW", "MID", "HIGH"], "parity": ["SAME", "DIFFERENT"], "vote": ["MINORITY", "MAJORITY"],
          "peak": ["P0", "P1", "P2", "P3"], "order": ["LESS", "NEAR", "GREATER"]}
DIM = {"alarm": 1, "zone": 2, "side": 2, "grade": 1, "parity": 2, "vote": 5, "peak": 4, "order": 2}
NOUN_SYN = dict(zip(NOUNS, ["furnace", "impeller", "faucet", "spinner", "reservoir", "wire", "strainer", "probe", "blower", "cog", "engine", "switch", "plunger", "bushing", "windmill", "warmer",
                            "refrigerator", "accumulator", "aerial", "reflector", "optic", "lodestone", "flexure", "retarder", "coupling", "jet", "meter", "breaker", "winding", "board", "auger", "turner",
                            "squeezer", "blender", "igniter", "freezer"]))
OOS = {"cannot_help": ["translate this paragraph into french", "write me a poem about the sea", "what is the capital of peru", "explain recursion simply"],
       "request_tool": ["send an email to the team", "open the calendar app"], "use_memory": ["remember my parking spot is level three", "where did i park my car"]}


def _expr(fam: str, p: dict) -> str:
    if fam == "alarm": return f"0 if x0 < {p['t']:.4f} else 1"
    if fam == "zone": return f"0 if (x0-{p['cx']:.4f})*(x0-{p['cx']:.4f}) + (x1-{p['cy']:.4f})*(x1-{p['cy']:.4f}) < {p['r2']:.4f} else 1"
    if fam == "side": return f"0 if {p['a']:.4f}*x0 + {p['b']:.4f}*x1 < {p['c']:.4f} else 1"
    if fam == "grade": return f"0 if x0 < {p['t1']:.4f} else 1 if x0 < {p['t2']:.4f} else 2"
    if fam == "parity": return f"0 if (x0 < {p['a']:.4f} and x1 < {p['b']:.4f}) or (x0 >= {p['a']:.4f} and x1 >= {p['b']:.4f}) else 1"
    if fam == "vote": return "1 if ((1 if x0 > 0.5 else 0) + (1 if x1 > 0.5 else 0) + (1 if x2 > 0.5 else 0) + (1 if x3 > 0.5 else 0) + (1 if x4 > 0.5 else 0)) >= " + str(p["m"]) + " else 0"
    if fam == "peak": return "0 if x0 >= max(x1, x2, x3) else 1 if x1 >= max(x0, x2, x3) else 2 if x2 >= max(x0, x1, x3) else 3"
    if fam == "order": return f"0 if x0 < x1 - {p['m']:.4f} else 2 if x0 > x1 + {p['m']:.4f} else 1"
    raise ValueError(fam)


def _params(fam: str, rng: random.Random) -> dict:
    if fam == "alarm": return {"t": rng.uniform(0.3, 0.6)}
    if fam == "zone": return {"cx": rng.uniform(-.2, .2), "cy": rng.uniform(-.2, .2), "r2": rng.uniform(0.3, 0.6)}
    if fam == "side": return {"a": rng.uniform(0.5, 1.5), "b": rng.choice([-1, 1]) * rng.uniform(0.5, 1.5), "c": rng.uniform(-.2, .2)}
    if fam == "grade": t1 = rng.uniform(0.25, 0.4); return {"t1": t1, "t2": t1 + rng.uniform(0.2, 0.35)}
    if fam == "parity": return {"a": rng.uniform(0.35, 0.65), "b": rng.uniform(0.35, 0.65)}
    if fam == "vote": return {"m": rng.choice([2, 3, 4])}
    if fam == "peak": return {}
    if fam == "order": return {"m": rng.uniform(0.05, 0.15)}
    raise ValueError(fam)


def _drifted(fam: str, p: dict) -> dict:
    """A rule change big enough to hurt: parameters move, the labels' meaning stays."""
    q = dict(p)
    if fam == "alarm": q["t"] = p["t"] + 0.25 if p["t"] < 0.5 else p["t"] - 0.25
    elif fam == "zone": q["r2"] = p["r2"] * 0.4; q["cx"] = p["cx"] + 0.3
    elif fam == "side": q["c"] = p["c"] + 0.5
    elif fam == "grade": q["t1"], q["t2"] = p["t1"] + 0.25, p["t2"] + 0.25
    elif fam == "parity": q["a"] = p["a"] + 0.25
    elif fam == "vote": q["m"] = 2 if p["m"] >= 3 else 4
    elif fam == "order": q["m"] = p["m"] + 0.3
    return q


@dataclass
class CapDef:
    cap_id: str
    noun: str
    family: str
    params: dict
    twin_of: str | None = None
    drifts: int = 0


class World:
    """Ground truth. 32 distinct capabilities (8 families x 4) plus 4 twins that implement exactly another capability's rule under a different name."""
    def __init__(self, seed: int = 0, n_caps: int = 36):
        rng = random.Random(seed); self.caps: list[CapDef] = []
        base = min(32, n_caps)
        for k in range(base):
            fam = FAMILIES[k % 8]; noun = NOUNS[k]
            self.caps.append(CapDef(f"{noun}_{fam}", noun, fam, _params(fam, rng)))
        for j, src in enumerate([0, 9, 18, 27][: max(0, n_caps - base)]):
            s = self.caps[src]; noun = NOUNS[32 + j]
            self.caps.append(CapDef(f"{noun}_{s.family}", noun, s.family, dict(s.params), twin_of=s.cap_id))
        self.by_id = {c.cap_id: c for c in self.caps}; self.by_noun = {c.noun: c for c in self.caps}
        self.by_token = {**self.by_noun, **{NOUN_SYN[c.noun]: c for c in self.caps}}

    def spec_dict(self, cap_id: str, n_train=None, n_val=None) -> dict:
        c = self.by_id[cap_id]; fam = c.family
        if n_train is None: n_train, n_val = (2500, 400) if fam == "vote" else (1000, 250)
        return {"capability_id": c.cap_id, "description": f"{c.noun} {fam} classifier", "keywords": [c.noun, fam], "labels": LABELS[fam], "input_dim": DIM[fam],
                "domain": [{"lo": -1.0 if fam in ("zone", "side") else 0.0, "hi": 1.0}] * DIM[fam], "label_expr": _expr(fam, c.params), "n_train": n_train, "n_val": n_val}

    def spec(self, cap_id: str, **kw) -> TaskSpec:
        d = self.spec_dict(cap_id, **kw); xs, ys = sample_spec(TaskSpec.from_dict({**d, "worked_examples": []}), 8, 99, unique=True)
        return TaskSpec.from_dict({**d, "worked_examples": [{"x": x, "y": y} for x, y in zip(xs, ys)]})

    def sample(self, cap_id: str, n: int, seed: int):
        return sample_spec(self.spec(cap_id), n, seed, unique=True)

    def label(self, cap_id: str, x) -> int:
        from training.safe_expr import compile_expr, evaluate, names_for
        c = self.by_id[cap_id]; return int(evaluate(compile_expr(_expr(c.family, c.params), names_for(DIM[c.family])), list(x)))

    def drift(self, cap_id: str):
        c = self.by_id[cap_id]; c.params = _drifted(c.family, c.params); c.drifts += 1

    def canonical(self, cap_id: str) -> str:
        c = self.by_id[cap_id]; return f"check the {c.noun} {c.family}"

    def paraphrases(self, cap_id: str) -> list[str]:
        """Two easy forms (the noun is kept) and two hard ones (noun and kind are both replaced by synonyms: no token overlaps the capability's keywords)."""
        c = self.by_id[cap_id]; s = SYN[c.family]; n2 = NOUN_SYN[c.noun]
        return [f"check the {c.noun} {s[0]}", f"run the {c.noun} {s[1]} test", f"check the {n2} {s[0]}", f"run the {n2} {s[1]} test"]

    def input_dim(self, cap_id: str) -> int: return DIM[self.by_id[cap_id].family]
    def labels(self, cap_id: str) -> list[str]: return LABELS[self.by_id[cap_id].family]


class BankTeacher(TeacherProvider):
    """Stands in for an LLM that knows the world as it is NOW (including drift) and what it cannot do. Counts calls and bytes."""
    name = "bank-teacher/1"

    def __init__(self, world: World, synonym_coverage: float = 0.0):
        """synonym_coverage p: with p > 0 each new-capability spec also lists alternative words, each of the capability's real synonyms included with probability p (deterministic per word),
        plus two distractor words borrowed from other capabilities (a model's synonym lists are neither complete nor clean). p = 0: no synonyms are offered."""
        self.w, self.calls, self.bytes_in, self.bytes_out = world, 0, 0, 0; self.p = synonym_coverage
        self.by_action: collections.Counter = collections.Counter()

    def _synonyms(self, cap) -> list[str]:
        import hashlib
        keep = lambda w: int(hashlib.sha256(f"{cap.cap_id}:{w}".encode()).hexdigest(), 16) % 1000 < self.p * 1000
        real = [w for w in (NOUN_SYN[cap.noun], *SYN[cap.family]) if keep(w)]
        others = [c.noun for c in self.w.caps if c.noun != cap.noun]; h = int(hashlib.sha256(cap.cap_id.encode()).hexdigest(), 16)
        return real + [others[h % len(others)], others[(h // 7) % len(others)]]

    def advise(self, req: HelpRequest) -> str:
        self.calls += 1; self.bytes_in += len(req.to_json())
        toks = set(req.task_intent.lower().replace(",", " ").split())
        cap = next((self.w.by_token[t] for t in toks if t in self.w.by_token), None)
        if cap is None:
            for action, phrases in OOS.items():
                if req.task_intent in phrases:
                    out = {"action": action, "rationale": "not a learnable capability"}
                    if action == "request_tool": out["tool"] = "send" if "send" in req.task_intent else "open"
                    if action == "use_memory": out["memory_text"] = req.task_intent
                    break
            else: out = {"action": "cannot_help", "rationale": "unknown request"}
        elif cap.cap_id in req.known_capabilities and "drift" not in req.failure:
            out = {"action": "reroute", "rationale": "an installed capability already solves this", "reroute_intent": self.w.canonical(cap.cap_id), "capability_id": cap.cap_id}
        else:
            d = self.w.spec_dict(cap.cap_id); xs, ys = self.w.sample(cap.cap_id, 6, 4242)
            out = {"action": "new_capability_spec", "rationale": "rule known to the teacher", "spec": {**d, "worked_examples": [{"x": x, "y": y} for x, y in zip(xs, ys)]}}
            if self.p > 0: out["spec"]["synonyms"] = self._synonyms(cap)
        s = json.dumps(out); self.bytes_out += len(s); self.by_action[out["action"]] += 1
        return s


# ---------------------------------------------------------------------------------------------- stream
def make_stream(world: World, n=720, seed=0, intro_start=6, intro_every=11, drift_at=((300, 0), (420, 9), (540, 33)), p_known=0.62, p_para=0.28, feedback_q=0.3) -> list[dict]:
    rng = random.Random(seed); caps = [c.cap_id for c in world.caps]
    intro_t = {caps[k]: (k if k < intro_start else intro_start + (k - intro_start) * intro_every) for k in range(len(caps))}
    drift_t = {t: caps[i] for t, i in drift_at}; events = []
    arrivals_per_t = {t: c for c, t in intro_t.items()}
    for t in range(n):
        introduced = [c for c in caps if intro_t[c] <= t]
        if t in drift_t: events.append({"t": t, "type": "drift", "cap": drift_t[t]})
        if t in arrivals_per_t: cap = arrivals_per_t[t]; typ = "new"
        else:
            w = [1.0 / (i + 1) ** 0.9 for i in range(len(introduced))]; cap = rng.choices(introduced, w)[0]
            r = rng.random(); typ = "known" if r < p_known else "paraphrase" if r < p_known + p_para else "oos"
        if typ == "oos":
            kind = rng.choice(list(OOS)); intent = rng.choice(OOS[kind]); events.append({"t": t, "type": "oos", "cap": None, "intent": intent, "x": [round(rng.random(), 3) for _ in range(rng.choice([1, 2, 4]))], "feedback": False}); continue
        intent = world.canonical(cap) if typ in ("known", "new") else rng.choice(world.paraphrases(cap))
        d = world.input_dim(cap); dom = -1.0 if world.by_id[cap].family in ("zone", "side") else 0.0
        events.append({"t": t, "type": typ, "cap": cap, "intent": intent, "x": [round(rng.uniform(dom, 1.0), 4) for _ in range(d)], "feedback": rng.random() < feedback_q})
    return events


class DriftMonitor:
    """Per-capability window of feedback outcomes; asks for a repair when a full-enough window shows low accuracy."""
    def __init__(self, window=12, min_samples=8, threshold=0.7):
        self.w, self.min, self.thr = window, min_samples, threshold; self.win: dict[str, collections.deque] = {}

    def add(self, cap: str, correct: bool) -> bool:
        d = self.win.setdefault(cap, collections.deque(maxlen=self.w)); d.append(bool(correct))
        return len(d) >= self.min and sum(d) / len(d) < self.thr

    def reset(self, cap: str): self.win.pop(cap, None)


class UncertainAsHelp:
    """Application policy: an ANSWER whose status is UNCERTAIN (ambiguous or weak route) is treated as NEEDS_HELP, i.e. escalated instead of served."""
    def __init__(self, cli): self._c = cli

    def solve(self, intent, x):
        r = self._c.solve(intent, x)
        if r.get("result") == "ANSWER" and r.get("status") == "UNCERTAIN":
            return {"result": "NEEDS_HELP", "reason_code": "UNCERTAIN_ROUTE", "reason": f"route to '{r.get('capability_id')}' is uncertain (score {r.get('route_score', 0):.2f})", "task_id": r.get("task_id")}
        return r

    def __getattr__(self, n): return getattr(self._c, n)


class CachingEscalator(Escalator):
    """Optional negative cache: exact intents the teacher said it cannot help with, that need an unavailable tool, or whose capability the build gates rejected are not asked again."""
    def __init__(self, *a, cache_refusals=False, **kw):
        super().__init__(*a, **kw); self.cache_refusals, self.refused = cache_refusals, set(); self.cache_hits = 0

    def solve(self, intent, x, examples=None):
        if self.cache_refusals and intent in self.refused:
            self.cache_hits += 1; rec = {"intent": intent, "path": ["local", "cached_refusal"], "teacher_called": False, "task_id": f"esc{len(self.log)}", "result": "NEEDS_HELP", "reason": "cached refusal"}
            self.log.append(rec); return rec
        rec = super().solve(intent, x, examples)
        if self.cache_refusals and rec.get("teacher_called") and (rec.get("teacher_action") in ("cannot_help", "request_tool") or
                (rec.get("teacher_action") == "new_capability_spec" and rec.get("result") == "NEEDS_HELP" and "rejected after" in (rec.get("reason") or ""))): self.refused.add(intent)
        return rec


# ---------------------------------------------------------------------------------------------- the modular system under test
class ModularRun:
    def __init__(self, tmp: Path, world: World, cache_refusals=False, router_updates=True, monitor=True, uncertain="escalate", seed=0, probe_rate=0.0, probe_k=3, strong_route=0.9, synonym_coverage=0.0, router=None, accept_synonyms=False):
        from packages.capbuild import keygen, write_trust
        from scripts.cli import Cli
        from training.service import BuildService
        self.tmp, self.world, self.opts = Path(tmp), world, dict(cache_refusals=cache_refusals, router_updates=router_updates, monitor=monitor, uncertain=uncertain, probe_rate=probe_rate)
        self.tmp.mkdir(parents=True, exist_ok=True)
        keygen("build-svc-1", self.tmp / "keys"); write_trust(self.tmp / "trust.json", {"build-svc-1": (self.tmp / "keys/build-svc-1.public").read_text()})
        self.teacher = BankTeacher(world, synonym_coverage)
        self.svc = BuildService(self.tmp / "keys", "build-svc-1", self.tmp / "build", self.teacher)
        self.cli = Cli(self.tmp / "rt", self.tmp / "trust.json", "PC_FULL")
        self.esc = CachingEscalator(UncertainAsHelp(self.cli) if uncertain == "escalate" else self.cli, self.teacher, self.svc, self.tmp / "esc", cache_refusals=cache_refusals)
        self.esc.persist_router_updates = router_updates; self.esc.accept_synonyms = accept_synonyms
        if router: self.cli.extra = ["--router", router]
        mk = {True: "window", "window": "window", "cusum": "cusum"}.get(monitor)
        self.mon_kind = mk; self.mon = {"window": DriftMonitor, "cusum": Cusum}[mk]() if mk else None
        self.probes = ProbeScheduler(probe_rate, probe_k, seed); self.strong_route = strong_route; self.probe_log: list[dict] = []
        self._ex: dict = {}; self.rows: list[dict] = []; self.repairs: list[dict] = []; self.checkpoints: list[dict] = []; self.build_seconds = 0.0; self.drift_times: dict[str, int] = {}

    # -- measurement against the world's CURRENT rules
    def accuracy_all(self, t: int, n=100) -> dict:
        out = {}
        for c in self.cli.list():
            cap = c["capability_id"]
            if cap not in self.world.by_id: continue
            xs, ys = self.world.sample(cap, n, 1000 + t)
            f = self.tmp / "eval.jsonl"; f.write_text("\n".join(json.dumps({"intent": "x", "input": x, "expected_index": y}) for x, y in zip(xs, ys)) + "\n")
            out[cap] = self.cli._run_detect("keyword", "batch", "--cases", str(f), capability=cap, )["summary"]["accuracy"]
        return out

    def stats(self) -> dict:
        caps = [c for c in self.cli.json("list-all") if c["role"] == "capability" and not c["archived"]]
        return {"active_modules": len(caps), "params": sum(c["params"] for c in caps), "model_bytes": sum(c["model_bytes"] for c in caps)}

    def _probe(self, cap: str, t: int):
        w = self.world; xs, ys = w.sample(cap, self.probes.k, 9000 + t)
        f = self.tmp / "probe.jsonl"; f.write_text("\n".join(json.dumps({"intent": "x", "input": x, "expected_index": y}) for x, y in zip(xs, ys)) + "\n")
        res = self.cli._run_detect("keyword", "batch", "--cases", str(f), capability=cap)["results"]
        wrong = [r.get("label_index") != y for r, y in zip(res, ys)]; self.probe_log.append({"t": t, "cap": cap, "wrong": sum(wrong)})
        alarm = False
        for wr in wrong:
            alarm = (self.mon.add(cap, not wr) if self.mon_kind == "window" else self.mon.add(cap, wr)) or alarm
        if alarm: self._repair(cap, t, xs[0])

    def _repair(self, cap: str, t: int, x):
        w = self.world; intent = w.canonical(cap)
        req = make_help_request(intent, w.input_dim(cap), list(self.esc.known), {"reason_code": "drift"}, "drift: feedback accuracy dropped")
        rec = {"intent": intent, "path": ["repair"], "teacher_called": True, "teacher_calls": 1, "task_id": f"repair{len(self.repairs)}"}
        self.esc.teacher_calls += 1
        t0 = time.time(); before = next(c["active_version"] for c in self.cli.list() if c["capability_id"] == cap)
        resp = parse_response(self.teacher.advise(req), self.esc.known)
        self.esc._act(resp, intent, x, w.sample(cap, 300, 5000 + t), rec)
        after = next(c["active_version"] for c in self.cli.list() if c["capability_id"] == cap)
        self.build_seconds += time.time() - t0
        self.repairs.append({"t": t, "cap": cap, "drift_t": self.drift_times.get(cap), "delay": t - self.drift_times[cap] if cap in self.drift_times else None, "result": rec.get("result"), "version_before": before, "version_after": after,
                             "reason": rec.get("reason"), "spurious": cap not in self.drift_times or w.by_id[cap].drifts == 0})
        if self.mon: self.mon.reset(cap)
        return rec

    def run(self, stream: list[dict], checkpoint_every=60, on_checkpoint=None):
        w = self.world
        for ev in stream:
            t = ev["t"]
            if ev["type"] == "drift":
                w.drift(ev["cap"]); self.drift_times[ev["cap"]] = t; continue
            cap = ev["cap"]; calls0 = self.teacher.calls; t0 = time.time()
            examples = None
            if cap is not None:                                                                                # the environment can always answer a verification query
                key = (cap, w.by_id[cap].drifts)
                if key not in self._ex: self._ex[key] = w.sample(cap, 200, 3000 + t)
                examples = self._ex[key]
            rec = self.esc.solve(ev["intent"], ev["x"], examples)
            if ev["type"] == "new": self.build_seconds += time.time() - t0
            served = rec.get("result") == "ANSWER"; correct = None
            if served and cap is not None:
                correct = rec.get("label") == w.labels(cap)[w.label(cap, ev["x"])]
            self.rows.append({"t": t, "type": ev["type"], "cap": cap, "served": served, "correct": correct, "status": rec.get("status"), "capability": rec.get("capability"), "teacher_calls": self.teacher.calls - calls0, "path": rec.get("path"), "result": rec.get("result"),
                              "learned": rec.get("learned"), "teacher_action": rec.get("teacher_action")})
            if self.mon and served and ev["feedback"] and correct is not None:
                # the CUSUM monitor only counts organic feedback whose route was near-exact (a routing mistake is not model drift); the window monitor counts everything (the Phase 14 original)
                counts = self.mon_kind == "window" or (rec.get("route_score") or 0) >= self.strong_route
                if counts and (self.mon.add(cap, correct) if self.mon_kind == "window" else self.mon.add(cap, not correct)): self._repair(cap, t, ev["x"])
            if self.mon and self.probes.rate:
                installed = [c["capability_id"] for c in self.cli.list() if c["capability_id"] in w.by_id]
                pc = self.probes.pick(t, installed)
                if pc is not None: self._probe(pc, t)
            if (t + 1) % checkpoint_every == 0:
                acc = self.accuracy_all(t); self.checkpoints.append({"t": t, "accuracy": acc, "mean_accuracy": float(np.mean(list(acc.values()))) if acc else None, "min_accuracy": min(acc.values()) if acc else None, **self.stats(), "teacher_calls": self.teacher.calls})
                if on_checkpoint: on_checkpoint(self, t)
        return self


# ---------------------------------------------------------------------------------------------- conventional baselines: ONE fixed-size network for everything
class Monolith:
    """Single MLP over [task one-hot | zero-padded features]; oracle task routing (an advantage the modular system does not get).
    mode: 'finetune' (train only on the new data), 'replay' (new data + a per-task buffer), 'joint' (retrain from scratch on everything seen)."""
    def __init__(self, caps: list[str], mode: str, hidden=128, epochs=60, buffer=150, seed=0, n_train=800):
        import torch, torch.nn as nn
        self.torch, self.nn = torch, nn
        self.caps, self.mode, self.epochs, self.buffer, self.seed, self.n_train = {c: i for i, c in enumerate(caps)}, mode, epochs, buffer, seed, n_train
        self.max_dim, self.max_cls = 5, 4; self.hidden = hidden
        self.data: dict[str, tuple] = {}; self.train_seconds = 0.0; self.events = 0
        torch.manual_seed(seed); self.net = self._new()

    def _new(self):
        nn = self.nn; return nn.Sequential(nn.Linear(len(self.caps) + self.max_dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.max_cls))

    def params(self) -> int: return sum(p.numel() for p in self.net.parameters())

    def _enc(self, cap, xs):
        t = self.torch; X = np.zeros((len(xs), len(self.caps) + self.max_dim), dtype=np.float32); X[:, self.caps[cap]] = 1.0
        for i, x in enumerate(xs): X[i, len(self.caps):len(self.caps) + len(x)] = x
        return X

    def learn(self, cap: str, xs, ys):
        t, nn = self.torch, self.nn; t0 = time.time(); self.events += 1
        xs, ys = xs[: self.n_train], ys[: self.n_train]
        if self.mode == "joint":
            self.data[cap] = (xs, ys); self.net = self._new(); sets = list(self.data.items())
        elif self.mode == "replay":
            sets = [(cap, (xs, ys))] + [(c, (a[: self.buffer], b[: self.buffer])) for c, (a, b) in self.data.items() if c != cap]; self.data[cap] = (xs, ys)
        else:
            sets = [(cap, (xs, ys))]; self.data[cap] = (xs, ys)
        X = np.vstack([self._enc(c, a) for c, (a, b) in sets]); Y = np.concatenate([np.asarray(b) for c, (a, b) in sets])
        X, Y = t.tensor(X), t.tensor(Y, dtype=t.long); opt = t.optim.Adam(self.net.parameters(), lr=3e-3); lossf = nn.CrossEntropyLoss()
        g = t.Generator().manual_seed(self.seed + self.events)
        for _ in range(self.epochs):
            perm = t.randperm(len(X), generator=g)
            for i in range(0, len(X), 128):
                b = perm[i:i + 128]; opt.zero_grad(); lossf(self.net(X[b]), Y[b]).backward(); opt.step()
        self.train_seconds += time.time() - t0

    def accuracy(self, cap: str, xs, ys) -> float:
        t = self.torch
        with t.no_grad(): p = self.net(t.tensor(self._enc(cap, xs))).argmax(1).numpy()
        return float((p == np.asarray(ys)).mean())
