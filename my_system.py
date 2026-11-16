"""Step 3: energy gate + local ASR (faster-whisper on the GPU) + the text rules in turn_rules.py.

While the agent talks:
  * the caller's sound is watched with the adaptive-floor energy gate (step 1);
  * when a burst of sound starts, a worker thread keeps transcribing the growing burst (latest audio only, never a
    queue) and turn_rules.decide() says STOP as soon as the words show the caller wants the floor;
  * if the ASR is not available (GPU libraries missing, model failed, repeated errors) the step-2a energy rule decides.
The 20 ms loop never waits for the ASR. TAKE_TURN (end of turn) is still the baseline rule.

    python -m ttbench.play                 # 8 calls in REAL time (~3.5 min): judge the ASR with this, NOT with --fast
    python -m ttbench.play --calls all     # all 24 calls in real time (~10 min)
    $env:ASR = "off"                       # force the energy-only fallback, to compare
    $env:ASR_MODEL = "small"               # a faster, weaker model
"""
import collections
import os
import sys
import threading
import time

import numpy as np

from turn_rules import decide
from ttbench.server import System, run

try:                                        # Arabic in log lines must never crash a thread on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TEAM = "Team 4 (for now :D)"                # your team name, shown on the scoreboard

SR = 16000
FRAME_MS = 20                               # each on_frame call covers 20 ms of audio
FRAME_SAMPLES = SR * FRAME_MS // 1000
LOUD_DB = -42                               # absolute minimum for "sound"
STOP_AFTER_MS = 200                         # (energy fallback) sound this long while the agent talks -> STOP
TAKE_TURN_AFTER_MS = 700                    # quiet this long after the caller spoke -> TAKE_TURN (baseline, unchanged)

# --- step 1: adaptive noise floor ---------------------------------------------------------------------------------
FLOOR_WINDOW_MS = 3000
FLOOR_PERCENTILE = 20
MARGIN_DB = 10                              # sound = this far above the recent floor

# --- step 2a: energy-only rule, now just the fallback ---------------------------------------------------------------
PEAK_DB = -20.0
GAP_TOLERANCE_FRAMES = 2
NO_PEAK = -120.0

# --- step 3: ASR ---------------------------------------------------------------------------------------------------
ASR_ON = os.environ.get("ASR", "on").lower() != "off"
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
ASR_COMPUTE = os.environ.get("ASR_COMPUTE", "float16")
HANG_MS = 300                               # this much quiet ends a burst (words inside a sentence have gaps)
FIRST_DECODE_MS = 300                       # do not decode a burst shorter than this
GROW_MS = 100                               # ask for a fresh decode once the burst has grown by this much
MAX_BURST_MS = 6000                         # decode at most this much of one burst
PAD_MS = 100                                # a little audio before the burst starts
RING_S = 10                                 # seconds of caller audio kept for the worker


def add_cuda_dlls():
    """Make the pip-installed CUDA 12 libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) findable on Windows."""
    try:
        import nvidia
    except ImportError:
        return
    for base in list(nvidia.__path__):
        for sub in ("cublas", "cudnn"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def loudness_db(samples: np.ndarray) -> float:
    """Loudness of 20 ms of 16-bit audio, in dB below full scale (silence is about -90, speech -40 to -15)."""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768
    return 20 * np.log10(rms + 1e-12)


class Asr:
    """Local Whisper on the GPU. One per process, loaded BEFORE the server starts (the harness wants a hello fast)."""

    def __init__(self):
        self.model = None
        self.ok = False

    def load(self):
        if not ASR_ON:
            print("[asr] off (ASR=off): energy rule only", flush=True)
            return
        add_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
            t0 = time.perf_counter()
            self.model = WhisperModel(ASR_MODEL, device="cuda", compute_type=ASR_COMPUTE)
            self.transcribe(np.zeros(SR, dtype=np.float32))     # warm-up: a missing CUDA library shows up here, not mid-call
            self.ok = True
            print(f"[asr] {ASR_MODEL} on cuda/{ASR_COMPUTE} ready in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as ex:                                  # no CPU fallback: ~3 s per decode is useless here
            self.ok = False
            print(f"[asr] NOT available ({ex!r}): energy rule only", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(audio, language="ar", beam_size=1, temperature=0.0,
                                        condition_on_previous_text=False, without_timestamps=True, max_new_tokens=24)
        return " ".join(s.text.strip() for s in segs)


ASR = Asr()


class MySystem(System):
    def __init__(self):
        self.agent_talking = False
        self.agent_text = ""
        self.stopped = False
        self.hist = collections.deque(maxlen=FLOOR_WINDOW_MS // FRAME_MS)     # recent caller levels (dB)
        # energy fallback (step 2a)
        self.sound_ms = 0
        self.gap = 0
        self.burst_peak = NO_PEAK
        # end of turn (baseline)
        self.quiet_ms = 0
        self.caller_spoke = False
        # ASR path: the 20 ms loop writes, one worker thread reads
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._ring = np.zeros(RING_S * SR, dtype=np.float32)     # last seconds of caller audio
        self._n = 0                                              # samples written so far (absolute)
        self._req = None                                         # newest decode request (older ones are dropped)
        self._epoch = 0                                          # changes when the agent utterance changes
        self._burst_id = 0
        self._in_burst = False
        self._burst_start = 0
        self._quiet = 0
        self._last_req_end = 0
        self._stop_flag = False
        self._fails = 0
        if ASR.ok:
            threading.Thread(target=self._worker, daemon=True).start()

    # --- what is happening in the call -------------------------------------------------------------------
    def on_scenario_start(self, context=None):
        self.hist.clear()                                        # new call, possibly new noise
        self._reset_burst()

    def on_agent_start(self, text, duration_ms):
        self.agent_talking, self.agent_text, self.stopped = True, text, False
        self.sound_ms, self.gap, self.burst_peak = 0, 0, NO_PEAK
        self._reset_burst()

    def on_agent_end(self):
        self.agent_talking, self.caller_spoke, self.quiet_ms = False, False, 0
        self._reset_burst()

    # --- every 20 ms: decide ---------------------------------------------------------------------------
    def on_frame(self, agent, caller):
        db = loudness_db(caller)
        self._push(caller)
        self.hist.append(db)
        floor = float(np.percentile(self.hist, FLOOR_PERCENTILE))
        loud_stop = db > max(LOUD_DB, floor + MARGIN_DB)      # adaptive: "a burst of sound exists"
        loud_eot = db > LOUD_DB                                # unchanged baseline, end of turn only

        if self.agent_talking:
            if self.stopped:
                return None
            if ASR.ok and self._fails < 3:
                self._track_burst(loud_stop)
                if self._stop_flag:
                    self.stopped = True
                    return "STOP"
                return None
            return self._energy_stop(loud_stop, db)           # ASR missing or failing: step-2a behaviour

        if loud_eot:
            self.caller_spoke, self.quiet_ms = True, 0
        elif self.caller_spoke:
            self.quiet_ms += FRAME_MS
            if self.quiet_ms >= TAKE_TURN_AFTER_MS:
                self.caller_spoke = False
                return "TAKE_TURN"
        return None

    # --- energy fallback ---------------------------------------------------------------------------------
    def _energy_stop(self, loud, db):
        if loud:
            self.sound_ms += FRAME_MS
            self.gap = 0
            self.burst_peak = max(self.burst_peak, db)
        else:
            self.gap += 1
            if self.gap >= GAP_TOLERANCE_FRAMES:
                self.sound_ms, self.burst_peak = 0, NO_PEAK
        if self.sound_ms >= STOP_AFTER_MS and self.burst_peak >= PEAK_DB:
            self.stopped = True
            return "STOP"
        return None

    # --- ASR path ----------------------------------------------------------------------------------------
    def _reset_burst(self):
        with self._lock:
            self._epoch += 1                                   # results for the old utterance are now stale
            self._in_burst = False
            self._quiet = 0
            self._req = None
            self._stop_flag = False

    def _push(self, caller):
        x = caller.astype(np.float32) / 32768.0
        with self._lock:
            self._ring[(self._n + np.arange(len(x))) % len(self._ring)] = x
            self._n += len(x)

    def _track_burst(self, sound):
        """Follow the caller's bursts of sound and ask the worker for a fresh decode as the burst grows."""
        if sound:
            self._quiet = 0
            if not self._in_burst:
                with self._lock:
                    self._burst_id += 1
                    self._in_burst = True
                    self._burst_start = self._n - FRAME_SAMPLES
                    self._last_req_end = 0
        elif self._in_burst:
            self._quiet += FRAME_MS
            if self._quiet >= HANG_MS:                         # burst over: one last look at all of it
                self._in_burst = False
                self._request()
                return
        if self._in_burst:
            ms = (self._n - self._burst_start) * 1000 // SR
            if (FIRST_DECODE_MS <= ms <= MAX_BURST_MS
                    and self._n - self._last_req_end >= GROW_MS * SR // 1000):
                self._request()

    def _request(self):
        with self._lock:
            self._req = (self._epoch, self._burst_id, self._burst_start, self._n, self.agent_text)
            self._last_req_end = self._n
        self._wake.set()

    def _snapshot(self, start, end):
        with self._lock:
            size = len(self._ring)
            lo = max(start - PAD_MS * SR // 1000, self._n - size, 0)
            hi = min(end, start + MAX_BURST_MS * SR // 1000)
            if hi - lo < SR // 10:
                return None
            return self._ring[np.arange(lo, hi) % size]        # fancy indexing makes a copy

    def _worker(self):
        while True:
            self._wake.wait()
            with self._lock:
                req, self._req = self._req, None
                self._wake.clear()
            if req is None:
                continue
            epoch, burst_id, start, end, agent_text = req
            audio = self._snapshot(start, end)
            if audio is None:
                continue
            try:
                t0 = time.perf_counter()
                text = ASR.transcribe(audio)
                ms = (time.perf_counter() - t0) * 1000
                self._fails = 0
            except Exception as ex:
                self._fails += 1
                print(f"[asr] decode failed ({ex!r}); {self._fails} in a row", flush=True)
                continue
            stop, why = decide(text, agent_text)
            print(f"[asr] burst {burst_id} {len(audio) / SR:.1f}s ({ms:.0f} ms): {text!r} -> "
                  f"{'STOP' if stop else 'keep'} ({why})", flush=True)
            if stop:
                with self._lock:
                    if epoch == self._epoch and self.agent_talking and not self.stopped:
                        self._stop_flag = True


if __name__ == "__main__":
    ASR.load()                                                 # heavy: before the server starts
    run(MySystem, TEAM, __file__)