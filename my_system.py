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

from turn_rules import decide, eot_incomplete
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
PAD_MS = 400                                # audio before the burst starts: with only 100 ms, Whisper turned short real speech
                                            # into «ترجمة نانسي قنقر» (6/27 stop pieces); with 400 ms, 0/27 and no new false stops
RING_S = 10                                 # seconds of caller audio kept for the worker
WARM = os.environ.get("ASR_WARM", "on").lower() != "off"     # keep the GPU awake between bursts: first decode 500 -> 270 ms
WARM_EVERY_S = 0.6
LOG_FILE = "asr_log.txt"                    # every decode, UTF-8: read this instead of the console (Arabic scrambles there)
AGENT_HISTORY = 3                           # echo check looks at the agent's last few sentences, not only the current one
REF_WINDOW_FRAMES = 10                      # 200 ms: the caller's reference voice must come from sustained sound

# --- end of turn, smart version (only when the ASR works; EOT=off gives the old fixed 700 ms rule) ------------------------
EOT_SMART = os.environ.get("EOT", "on").lower() != "off"
EOT_PROBE_MS = 300                          # caller quiet this long -> decode what they just said
EOT_WINDOW_MS = 4000                        # ...the last 4 s of their audio (only the ending matters)
EOT_WAIT_RESULT_MS = 900                    # no answer from the ASR by then -> take the turn anyway
EOT_HOLD_MS = 1600                          # text ends mid-thought: wait this long (total quiet) before taking the turn


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
            self.transcribe(np.zeros(SR, dtype=np.float32))[0]     # warm-up: a missing CUDA library shows up here, not mid-call
            self.ok = True
            print(f"[asr] {ASR_MODEL} on cuda/{ASR_COMPUTE} ready in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as ex:                                  # no CPU fallback: ~3 s per decode is useless here
            self.ok = False
            print(f"[asr] NOT available ({ex!r}): energy rule only", flush=True)

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 24):
        """-> (text, mean avg_logprob of the segments): Whisper's own confidence, logged to study garbled decodes."""
        segs, _ = self.model.transcribe(audio, language="ar", beam_size=1, temperature=0.0,
                                        condition_on_previous_text=False, without_timestamps=True,
                                        max_new_tokens=max_new_tokens)
        segs = list(segs)
        logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else 0.0
        return " ".join(s.text.strip() for s in segs), logprob


ASR = Asr()


class MySystem(System):
    def __init__(self):
        self.agent_talking = False
        self.agent_text = ""
        self.agent_hist = []
        self.stopped = False
        self.hist = collections.deque(maxlen=FLOOR_WINDOW_MS // FRAME_MS)     # recent caller levels (dB)
        self._ref_db = NO_PEAK                                   # caller's loudest 20 ms this call, outside bursts being judged
        self._burst_ref = None
        self._ref_win = collections.deque(maxlen=REF_WINDOW_FRAMES)   # (dB, loud?) of the last 200 ms
        # energy fallback (step 2a)
        self.sound_ms = 0
        self.gap = 0
        self.burst_peak = NO_PEAK
        # end of turn (baseline)
        self.quiet_ms = 0
        self.caller_spoke = False
        self._eot_id = 0                                         # changes whenever the caller makes sound again
        self._eot_probed = False
        self._eot_req = None
        self._eot_res = None                                     # (eot_id, incomplete?, reason) for the current silence
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
        self.agent_hist = []
        self._ref_db = NO_PEAK                                   # new caller, new line: no reference voice yet
        self._ref_win.clear()
        self._log(f"--- scenario {context!r}"[:200])
        self._reset_burst()

    def on_agent_start(self, text, duration_ms):
        self.agent_talking, self.agent_text, self.stopped = True, text, False
        self.agent_hist = (self.agent_hist + [text])[-AGENT_HISTORY:]
        self.sound_ms, self.gap, self.burst_peak = 0, 0, NO_PEAK
        self._reset_burst()

    def on_agent_end(self):
        # We stopped because the caller spoke: they have the floor even if they already went quiet (a short «لا»
        # can be over before the STOP lands), so the end-of-turn clock starts now instead of waiting for new sound.
        took_floor = self.stopped
        self.agent_talking, self.caller_spoke, self.quiet_ms = False, took_floor, 0
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
                if not self._in_burst:                         # a burst never sets its own reference
                    self._update_ref(db, loud_stop)
                if self._stop_flag:
                    self.stopped = True
                    return "STOP"
                return None
            return self._energy_stop(loud_stop, db)           # ASR missing or failing: step-2a behaviour

        smart = EOT_SMART and ASR.ok and self._fails < 3
        self._update_ref(db, loud_stop)                        # the agent is quiet: this is the caller's own turn
        if loud_eot:
            self.caller_spoke, self.quiet_ms = True, 0
            self._eot_id += 1                                  # any probe in flight is now out of date
            self._eot_probed, self._eot_res = False, None
        elif self.caller_spoke:
            self.quiet_ms += FRAME_MS
            if smart and not self._eot_probed and self.quiet_ms >= EOT_PROBE_MS:
                self._eot_probed = True
                self._request_eot()
            if self.quiet_ms >= TAKE_TURN_AFTER_MS and self._eot_ready(smart):
                self.caller_spoke = False
                return "TAKE_TURN"
        return None

    def _update_ref(self, db, loud):
        """The caller's reference voice = the loudest frame of SUSTAINED sound: at least half of the last 200 ms is
        loud. Speech always qualifies; a click or a bump on the mic (a few frames) never sets the reference."""
        self._ref_win.append((db, loud))
        if sum(1 for _, l in self._ref_win) >= REF_WINDOW_FRAMES and \
                sum(1 for _, l in self._ref_win if l) * 2 >= REF_WINDOW_FRAMES:
            self._ref_db = max(self._ref_db, max(d for d, _ in self._ref_win))

    def _eot_ready(self, smart):
        """Quiet for 700 ms already. Take the turn now, or is the caller probably still thinking?"""
        if not smart:
            return True
        res = self._eot_res
        if res is None:                                        # the ASR has not answered yet
            return self.quiet_ms >= EOT_WAIT_RESULT_MS
        if res[1]:                                             # text ends mid-thought: give them more time
            return self.quiet_ms >= EOT_HOLD_MS
        return True

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
            self._eot_id += 1                                  # a new agent turn: old end-of-turn probes are stale
            self._eot_probed, self._eot_res, self._eot_req = False, None, None

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
                    self._burst_ref = self._ref_db if self._ref_db > LOUD_DB else None   # no voice heard yet: absolute gate
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
            self._req = (self._epoch, self._burst_id, self._burst_start, self._n, " ".join(self.agent_hist),
                         self._burst_ref)
            self._last_req_end = self._n
        self._wake.set()

    def _request_eot(self):
        with self._lock:
            self._eot_req = (self._eot_id, self._n - EOT_WINDOW_MS * SR // 1000, self._n)
        self._wake.set()

    def _do_eot(self, ereq):
        eot_id, start, end = ereq
        audio = self._snapshot(max(start, 0), end)
        if audio is None:
            return
        try:
            t0 = time.perf_counter()
            text, _ = ASR.transcribe(audio, max_new_tokens=64)
            ms = (time.perf_counter() - t0) * 1000
        except Exception as ex:
            self._fails += 1
            print(f"[asr] eot decode failed ({ex!r}); {self._fails} in a row", flush=True)
            return
        inc, why = eot_incomplete(text)
        line = f"[eot] ({ms:.0f} ms): {text!r} -> {'WAIT' if inc else 'take turn'} ({why})"
        print(line, flush=True)
        self._log(line)
        if eot_id == self._eot_id:                             # the caller has not made sound since
            self._eot_res = (eot_id, inc, why)

    def _snapshot(self, start, end):
        with self._lock:
            size = len(self._ring)
            lo = max(start - PAD_MS * SR // 1000, self._n - size, 0)
            hi = min(end, start + MAX_BURST_MS * SR // 1000)
            if hi - lo < SR // 10:
                return None
            return self._ring[np.arange(lo, hi) % size]        # fancy indexing makes a copy

    def _log(self, line):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _worker(self):
        silence = np.zeros(SR, dtype=np.float32)
        while True:
            woke = self._wake.wait(timeout=WARM_EVERY_S if WARM else None)
            if not woke:                                       # WARM experiment: nothing to do, keep the GPU awake
                if not self._in_burst and (self.agent_talking or (self.caller_spoke and EOT_SMART)):
                    try:
                        ASR.transcribe(silence)
                    except Exception:
                        pass
                continue
            with self._lock:
                req, self._req = self._req, None
                ereq, self._eot_req = self._eot_req, None
                self._wake.clear()
            if ereq is not None:
                self._do_eot(ereq)
            if req is None:
                continue
            epoch, burst_id, start, end, agent_text, ref = req
            audio = self._snapshot(start, end)
            if audio is None:
                continue
            try:
                t0 = time.perf_counter()
                text, logprob = ASR.transcribe(audio)
                ms = (time.perf_counter() - t0) * 1000
                self._fails = 0
            except Exception as ex:
                self._fails += 1
                print(f"[asr] decode failed ({ex!r}); {self._fails} in a row", flush=True)
                continue
            body = audio[min(PAD_MS * SR // 1000, max(0, len(audio) - FRAME_SAMPLES)):]   # loudness: the burst only, not the lead-in
            fr = body[: len(body) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
            pk = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1)).max() + 1e-12) if len(fr) else -99.0
            stop, why = decide(text, agent_text, peak_db=float(pk), ref_db=ref)
            rel = f"{pk - ref:+.0f}" if ref is not None else "n/a"
            line = (f"[asr] burst {burst_id} {len(audio) / SR:.1f}s ({ms:.0f} ms, lp {logprob:.2f}, pk {pk:.0f} dB, "
                    f"vs caller {rel} dB): {text!r} -> "
                    f"{'STOP' if stop else 'keep'} ({why})")
            print(line, flush=True)
            self._log(line)
            if stop:
                with self._lock:
                    if epoch == self._epoch and self.agent_talking and not self.stopped:
                        self._stop_flag = True


if __name__ == "__main__":
    ASR.load()                                                 # heavy: before the server starts
    run(MySystem, TEAM, __file__)