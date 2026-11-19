"""Robustness tests for my_system.py (no GPU needed): python test_asr_robustness.py   (or pytest -s)

1. three+ Whisper failures -> energy-rule fallback -> Whisper comes back after the retry (model reloaded too)
2. server started with Whisper unavailable -> loud warning, keeps running, answers hello + ping
3. exceptions / a dead worker thread never break the 20 ms loop
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import my_system as ms                                     # noqa: E402

FRAME = ms.FRAME_SAMPLES


def quiet():
    return (np.random.randn(FRAME) * 5).astype(np.int16)           # about -76 dB


def loud():
    t = np.arange(FRAME) / ms.SR
    return (0.2 * 32768 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)    # about -17 dB: passes the -20 dB fallback gate


def run_frames(system, n, caller, realtime=True):
    """-> list of (frame index, decision) for every non-None decision. Real time so the worker thread keeps pace."""
    out = []
    zeros = np.zeros(FRAME, dtype=np.int16)
    for i in range(n):
        d = system.on_frame(zeros, caller() if callable(caller) else caller)
        if d:
            out.append((i, d))
        if realtime:
            time.sleep(ms.FRAME_MS / 1000)
    return out


def new_system(monkeypatch=None):
    s = ms.MySystem()
    s.on_scenario_start({"use_case": "test"})
    return s


def agent_turn(s):
    s.on_agent_start("مرحبا كيف أقدر أساعدك اليوم", 5000)


def test_failures_then_recovery():
    print("\n=== TEST 1: 3+ consecutive Whisper failures -> fallback -> recovery ===")
    calls = {"n": 0, "fail": True, "loads": 0}

    def fake_transcribe(audio, max_new_tokens=24):
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("CUDA error: simulated")
        return "", 0.0                                             # silence: nothing heard -> decide() keeps

    def fake_load_model():
        calls["loads"] += 1
        ms.ASR.model = object()

    saved = (ms.ASR.transcribe, ms.ASR._load_model, ms.ASR.model, ms.ASR.ok, ms.RETRY_S, ms.ASR._last_try)
    ms.ASR.transcribe, ms.ASR._load_model = fake_transcribe, fake_load_model
    ms.ASR.model, ms.ASR.ok, ms.RETRY_S, ms.ASR._last_try = object(), True, 1.5, time.monotonic()
    ms.WARM = False
    try:
        s = new_system()
        print(f"start: live={s._asr_live}  fails={s._fails}")
        agent_turn(s)
        run_frames(s, 60, quiet)                                   # noise floor
        # (a) Whisper failing: bursts get decoded, decodes fail, after 3 the energy rule must take over and STOP
        decisions = run_frames(s, 100, loud)
        print(f"during failures: fails={s._fails} live={s._asr_live} decisions={decisions}")
        assert s._fails >= 3 and not s._asr_live, "should have fallen back after 3 failures"
        assert decisions and decisions[0][1] == "STOP", "fallback energy rule must still STOP on loud caller sound"
        s.on_agent_end()
        run_frames(s, 20, quiet)
        # (b) Whisper repaired; the retry (every 1.5 s here) must notice by itself
        calls["fail"] = False
        t0 = time.monotonic()
        while not s._asr_live and time.monotonic() - t0 < 8:
            run_frames(s, 5, quiet)
        print(f"recovery after {time.monotonic() - t0:.1f}s: live={s._asr_live} fails={s._fails} model reloads={calls['loads']}")
        assert s._asr_live and s._fails == 0, "Whisper should be back"
        assert calls["loads"] >= 1, "a failed retry must have dropped and reloaded the model"
        # (c) Whisper really in use again: a loud burst is decoded and (empty text) does NOT trigger the energy-rule STOP
        n0 = calls["n"]
        agent_turn(s)
        run_frames(s, 60, quiet)
        decisions = run_frames(s, 60, loud)
        print(f"after recovery: transcribe calls +{calls['n'] - n0}, decisions={decisions}")
        assert calls["n"] > n0, "Whisper is not being called again"
        assert not decisions, "with Whisper back, an empty transcript must keep (no energy-rule STOP)"
        print("TEST 1 PASSED")
    finally:
        ms.ASR.transcribe, ms.ASR._load_model, ms.ASR.model, ms.ASR.ok, ms.RETRY_S, ms.ASR._last_try = saved
        ms.WARM = os.environ.get("ASR_WARM", "on").lower() != "off"


def test_worker_exceptions_and_death():
    print("\n=== TEST 3: worker exception / worker death never break the 20 ms loop ===")

    class Boom(BaseException):
        pass

    saved = (ms.ASR.transcribe, ms.ASR.model, ms.ASR.ok, ms.decide)
    ms.ASR.transcribe = lambda audio, max_new_tokens=24: ("", 0.0)
    ms.ASR.model, ms.ASR.ok = object(), True
    try:
        s = new_system()
        agent_turn(s)
        run_frames(s, 60, quiet)
        # decide() raising inside the worker: logged, thread survives
        def bad_decide(*a, **k):
            raise ValueError("bug in a rule")
        ms.decide = bad_decide
        t0 = time.perf_counter()
        run_frames(s, 40, loud)
        worst = 0.0
        print(f"decide() raised inside the worker; thread alive={s._thread.is_alive()}")
        assert s._thread.is_alive()
        ms.decide = saved[3]
        # the thread dying completely (BaseException): the loop restarts it within ~1 s
        orig_step = s._worker_step
        state = {"done": False}

        def dying(silence):
            if not state["done"]:
                state["done"] = True
                raise Boom()
            return orig_step(silence)
        s._worker_step = dying
        s._wake.set()
        time.sleep(0.2)
        old = s._thread
        print(f"after BaseException: old thread alive={old.is_alive()}")
        assert not old.is_alive()
        lat = []
        zeros = np.zeros(FRAME, dtype=np.int16)
        for _ in range(80):
            t = time.perf_counter()
            s.on_frame(zeros, quiet())
            lat.append(time.perf_counter() - t)
            time.sleep(ms.FRAME_MS / 1000)
        print(f"new thread alive={s._thread.is_alive()} (restarted: {s._thread is not old}); "
              f"worst on_frame time while all that happened: {max(lat) * 1000:.2f} ms")
        assert s._thread.is_alive() and s._thread is not old
        assert max(lat) < 0.02, "on_frame must stay far below 20 ms"
        print("TEST 3 PASSED")
    finally:
        ms.ASR.transcribe, ms.ASR.model, ms.ASR.ok, ms.decide = saved


def test_server_without_whisper():
    print("\n=== TEST 2: server started with Whisper unavailable ===")
    import websockets
    port = 8791
    code = ("import sys, runpy; sys.modules['faster_whisper'] = None; "          # `import faster_whisper` now raises ImportError
            f"sys.argv = ['my_system.py', '--port', '{port}']; runpy.run_path('my_system.py', run_name='__main__')")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log0 = os.path.getsize(os.path.join(HERE, ms.LOG_FILE)) if os.path.exists(os.path.join(HERE, ms.LOG_FILE)) else 0
    p = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    lines, ok = [], False
    try:
        t0 = time.time()
        while time.time() - t0 < 60:
            ln = p.stdout.readline()
            if not ln:
                break
            lines.append(ln.rstrip())
            print("  | " + lines[-1])
            if "listening on" in ln:
                ok = True
                break
        assert ok, "server did not start"
        text = "\n".join(lines)
        assert "WHISPER IS NOT AVAILABLE" in text and "!!!!!!!!" in text, "loud warning missing"

        async def talk():
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                hello = json.loads(await ws.recv())
                await ws.send(json.dumps({"type": "scenario_start", "scenario_id": "t", "context": {}}))
                await ws.send(json.dumps({"type": "agent_utterance_start", "text": "مرحبا", "duration_ms": 3000}))
                t = time.perf_counter()
                await ws.send(json.dumps({"type": "ping", "id": 7}))
                pong = json.loads(await ws.recv())
                # loud caller frames: the energy fallback must still STOP
                got = None
                for i in range(40):
                    c = (quiet() if i < 15 else loud())
                    fr = np.stack([np.zeros(FRAME, dtype=np.int16), c], axis=1).astype("<i2").tobytes()
                    await ws.send(fr)
                    try:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 0.03))
                        if m.get("decision"):
                            got = m["decision"]
                            break
                    except asyncio.TimeoutError:
                        pass
                return hello, pong, got, (time.perf_counter() - t) * 1000

        hello, pong, got, ms_ = asyncio.run(talk())
        print(f"hello team={hello['team']!r}; pong id={pong['id']}; fallback decision on loud caller: {got}")
        assert pong["id"] == 7 and got == "STOP"
        time.sleep(0.3)
        log = open(os.path.join(HERE, ms.LOG_FILE), "rb").read()[log0:].decode("utf-8", "replace")
        assert "WHISPER IS NOT AVAILABLE" in log, "warning not written to asr_log.txt"
        print("warning also in asr_log.txt: yes")
        print("TEST 2 PASSED")
    finally:
        p.terminate()
        try:
            p.wait(5)
        except Exception:
            p.kill()


if __name__ == "__main__":
    test_failures_then_recovery()
    test_server_without_whisper()
    test_worker_exceptions_and_death()
    print("\nALL PASSED")
