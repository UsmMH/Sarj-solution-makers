"""Virtual-clock replay of a benchmark split: the real system, the real Whisper, but no waiting for the wall clock.

    python sim.py --data data/dev --name sim-dev                    # -> runs/sim-dev/{log.jsonl, run.json, score.json, report.md}
    python sim.py --data data/split_train --name sim-train --system my_system_cand

How it stays honest to the real-time harness:
  * frames go through MySystem.on_frame exactly as the server feeds them (controls of frame n before frame n);
  * the worker thread is replaced by a virtual worker: when it is free and a request is waiting it takes it (same code:
    _eot_compute / _decode), the decode runs on the GPU now, and its result lands (_eot_apply / _apply) only once the
    virtual clock has moved on by the decode's measured time. While busy, newer requests replace older ones, as live;
  * the warm-up decodes the live worker makes every 0.6 s of idle time also keep the virtual worker busy;
  * decisions are stamped at frame time + 2 ms (the live round trip is ~1-3 ms) and scored by ttbench.score.
Decodes are cached by the exact audio they read (analysis/sim_cache/), with their measured time, so a rule-only change
replays a whole split in seconds and gives the same answer every time (no run-to-run noise). Change the model or
anything that changes the audio windows and the new windows are decoded for real.
Not modelled: thread-scheduling jitter and GPU contention, which is why final candidates still get a real-time run.
"""
import argparse
import hashlib
import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from ttbench.dataset import lay_out_stream, load_split
from ttbench.harness import build_stream
from ttbench.protocol import FRAME_MS, FRAME_SAMPLES
from ttbench.score import score_run_dir

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RTT_MS = 2.0
SLOW = float(os.environ.get("SIM_SLOW", "1"))   # stress test: every decode takes this many times longer (busy / slow GPU)
OUTLIER_MS = 1500.0                 # a cached decode slower than this is re-measured


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.d = pickle.loads(path.read_bytes()) if path.exists() else {}
        self.new = 0

    def save(self):
        if self.new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(self.d))
            os.replace(tmp, self.path)
            self.new = 0


STATE = {"ms": 0.0}
_EXTRA = []                         # caches of extra models (FIRST_LOOK)                 # virtual decode time accumulated by every wrapped model since the last reset


def install_cache(ms_mod, cache: Cache, tag: str, asr=None):
    """Wrap asr.transcribe: answers come from the cache when this exact audio was decoded before."""
    asr = asr or ms_mod.ASR
    if getattr(asr, "_sim_state", None) is not None:              # already wrapped (several runs in one process)
        return asr._sim_state
    real = asr.transcribe
    state = STATE

    def transcribe(audio, max_new_tokens=24):
        key = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest() + f":{max_new_tokens}:{tag}"
        hit = cache.d.get(key)
        if hit is None or hit[3] > OUTLIER_MS:                    # never trust a decode time from a cold / busy GPU
            t0 = time.perf_counter()
            text, lp = real(audio, max_new_tokens)
            ms = (time.perf_counter() - t0) * 1000
            hit = (text, lp, dict(getattr(asr, "last_info", {}) or {}), ms)
            cache.d[key] = hit
            cache.new += 1
            if cache.new % 200 == 0:
                cache.save()
        asr.last_info = hit[2]
        asr.last_ms = hit[3] * SLOW
        state["ms"] += hit[3] * SLOW
        return hit[0], hit[1]

    asr.transcribe = transcribe
    asr._sim_state = state
    return state


def simulate(ms_mod, data: Path, out: Path, cache: Cache, tag: str, quiet=True):
    state = install_cache(ms_mod, cache, tag)

    class Sim(ms_mod.MySystem):
        def _start_worker(self):
            pass

        def _ensure_worker(self):
            pass

    if not quiet:
        pass
    scen = lay_out_stream(load_split(data))
    stream, controls = build_stream(scen)
    n_frames = len(stream) // FRAME_SAMPLES
    sysm = Sim()
    log = []
    seq = 0
    busy_until = 0.0
    idle_since = 0.0
    pending = []                                                   # (t_apply, fn, arg)
    warm_ms = None
    for n in range(n_frames):
        t = n * FRAME_MS
        # results that have landed by now (the live worker sets flags between frames)
        due = [p for p in pending if p[0] <= t]
        pending = [p for p in pending if p[0] > t]
        for _, fn, arg in sorted(due, key=lambda p: p[0]):
            fn(arg)
        for msg in controls.get(n, []):
            seq += 1
            log.append({"seq": seq, "dir": "tx", "stream_ms": t, "msg": msg})
            ty = msg["type"]
            if ty == "scenario_start":
                sysm.on_scenario_start(msg.get("context", {}))
            elif ty == "agent_utterance_start":
                sysm.on_agent_start(msg.get("text", ""), msg.get("duration_ms", 0))
            elif ty == "agent_utterance_end":
                sysm.on_agent_end()
        frame = stream[n * FRAME_SAMPLES:(n + 1) * FRAME_SAMPLES]
        out_d = sysm.on_frame(frame[:, 0], frame[:, 1])
        if out_d is not None:
            for d in (out_d if isinstance(out_d, list) else [out_d]):
                m = {"decision": d} if isinstance(d, str) else dict(d)
                seq += 1
                log.append({"seq": seq, "dir": "rx", "stream_ms": t + RTT_MS, "msg": {"type": "decision", **m}})
        # virtual worker (wakes as soon as a request exists and it is free)
        if busy_until > t or not sysm._asr_live:
            continue
        with sysm._lock:
            req, sysm._req = sysm._req, None
            ereq, sysm._eot_req = sysm._eot_req, None
        if req is None and ereq is None:
            if ms_mod.WARM and t - idle_since >= ms_mod.WARM_EVERY_S * 1000 and not sysm._in_burst and \
                    (sysm.agent_talking or (sysm.caller_spoke and ms_mod.EOT_SMART)):
                if warm_ms is None:
                    state["ms"] = 0.0
                    ms_mod.ASR.transcribe(np.zeros(ms_mod.SR, dtype=np.float32))
                    warm_ms = state["ms"]
                busy_until = t + warm_ms
                idle_since = busy_until
            continue
        clock = t
        if ereq is not None:
            state["ms"] = 0.0
            r = sysm._eot_compute(ereq)
            clock += state["ms"] if r is not None else 0
            if r is not None:
                pending.append((clock, sysm._eot_apply, r))
        if req is not None:
            state["ms"] = 0.0
            r = sysm._decode(req)
            clock += state["ms"]
            if r is not None:
                pending.append((clock, sysm._apply, r))
        busy_until = clock
        idle_since = clock
    cache.save()
    for c in _EXTRA:
        c.save()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.jsonl", "w", encoding="utf-8") as f:
        for line in log:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    meta = {"harness_version": "sim", "speed": 1.0, "official": True, "completed": True, "errors": [],
            "hello": {"team": ms_mod.TEAM, "system_version": f"sim:{ms_mod.__name__}"},
            "scenarios": [{"scenario_id": s.scenario_id, "opaque_id": s.opaque_id, "start_ms": s.start_ms,
                           "duration_ms": s.duration_ms} for s in scen]}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return score_run_dir(data, out)


def load_system(name):
    return importlib.import_module(name)


def prepare(mod):
    """Load + warm Whisper once; -> (cache, tag)."""
    mod.ASR.load()
    if not mod.ASR_ON:
        return Cache(Path("analysis/sim_cache/none.pkl")), "off"
    if not mod.ASR.ok:
        raise SystemExit("Whisper did not load: is the venv active?")
    for _ in range(5):                                          # warm the GPU: the live worker keeps it warm too
        mod.ASR.transcribe(np.random.default_rng(0).normal(0, 0.01, mod.SR).astype(np.float32))
    tag = f"{mod.ASR_MODEL}|{mod.ASR_COMPUTE}"
    fl = getattr(mod, "FIRST_LOOK", None)
    if fl is not None:
        fl.load()
        for _ in range(5):
            fl.transcribe(np.random.default_rng(0).normal(0, 0.01, mod.SR).astype(np.float32))
        ftag = f"{fl.model_name}|{mod.ASR_COMPUTE}"
        fc = Cache(Path("analysis/sim_cache") / (hashlib.sha1(ftag.encode()).hexdigest()[:10] + ".pkl"))
        install_cache(mod, fc, ftag, fl)
        _EXTRA.append(fc)
    return Cache(Path("analysis/sim_cache") / (hashlib.sha1(tag.encode()).hexdigest()[:10] + ".pkl")), tag


def summary(res: dict) -> str:
    c, lat, r = res["core"], res["latency_ms"], res["rates"]
    return (f"Core {c['core_score']:.1f}  R {c['R']:.2f}  S {c['S']:.2f}  lat med {lat['median']} p90 {lat['p90']}  "
            f"false stops {r['false_stops']}  misses {r['misses']}/{r['n_stop_events']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--system", default="my_system", help="module name of the system file (default my_system)")
    a = p.parse_args()
    if "split_test" in str(a.data) and os.environ.get("UNLOCK_TEST") != "yes-final":
        raise SystemExit("data/split_test is LOCKED: set UNLOCK_TEST=yes-final for the single final run")
    mod = load_system(a.system)
    mod.LOG_FILE = str(Path("runs") / a.name / "asr_log.txt")
    Path(mod.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(mod.LOG_FILE).write_text("", encoding="utf-8")
    import builtins
    _print = builtins.print
    builtins.print = lambda *x, **k: None if (x and isinstance(x[0], str) and x[0].startswith(("[asr] burst", "[eot]"))) else _print(*x, **k)
    cache, tag = prepare(mod)
    res = simulate(mod, a.data, Path("runs") / a.name, cache, tag)
    builtins.print = _print
    print(f"{a.name}: {summary(res)}")


if __name__ == "__main__":
    main()
