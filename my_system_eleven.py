"""Team 4 SUBMITTED system (Nov 20 freeze): ElevenLabs live speech-to-text (scribe_v3_live) instead of local Whisper.
Start it with `python my_system_eleven.py --host 0.0.0.0` (my_system.py = the Whisper alternative, which this file builds
on). Everything except the recogniser is the same code as my_system.py:
adaptive floor, burst tracking, the caller's reference voice, turn_rules.decide(), speculative STOP + RESUME, energy fallback.

How it differs:
- The caller channel is streamed to ElevenLabs all the time (100 ms messages). Every partial transcript that arrives while
  the agent talks and a caller burst is open is judged at once by decide(). No 300 ms wait, no decode queue, no GPU.
- When a burst ends (300 ms quiet) or a new agent utterance starts, we commit: the next burst starts with fresh text.
- End of turn: at 300 ms of caller quiet we commit and run eot_incomplete() on the committed text (Whisper did this before).
- ElevenLabs down or not reachable -> the energy rule, as my_system.py does when Whisper fails; it reconnects by itself.

Run (no GPU needed; needs ELEVENLABS_API_KEY in .env and internet):
    python my_system_eleven.py
    python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --name eleven-dev
Decisions are logged to eleven_log.txt. It is a CLOUD system: its network delay counts in the score and must be declared.
"""
import asyncio
import base64
import collections
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

import my_system as M
from my_system import MySystem, loudness_db, loud_warning

TEAM = "Team 4 (for now :D)"                # your team name, shown on the scoreboard
from ttbench.server import run
from turn_rules import NEUTRAL, decide, eot_incomplete, is_ack, tokens

URL = ("wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v3_live&audio_format=pcm_16000"
       "&language_code=ar&commit_strategy=manual")
CHUNK_FRAMES = max(1, int(os.environ.get("ELEVEN_CHUNK_MS", "100")) // M.FRAME_MS)   # audio per websocket message
MIN_COMMIT_FRAMES = 16        # ElevenLabs closes the session on a commit with < 0.3 s of new audio: wait for 0.32 s
COMMIT_TIMEOUT_S = 1.5        # a commit with no answer by then is given up (partials are ignored while one is pending)
LOG_FILE = "eleven_log.txt"
HERE = Path(__file__).resolve().parent


def _env(name):
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        for line in (HERE / ".env").read_text(encoding="utf-8-sig").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip("'\"")
    except OSError:
        pass
    return None


def _log(line):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class Eleven:
    """One live Scribe connection for the whole process, in its own thread with its own asyncio loop. The 20 ms loop only
    queues audio / commits (never waits). Results go to `target` (the system of the current harness session)."""

    def __init__(self):
        self.ok = False
        self.target = None
        self.loop = None
        self.q = None
        self._thread = None
        self._sent = 0                 # commits sent on this connection
        self._recv = 0                 # committed transcripts received
        self._cbs = {}                 # commit number -> [callback(text), ...]
        self._want = None              # commits asked for but not sent yet (too little new audio): [callbacks]
        self._since = 0                # frames sent since the last commit
        self._commit_t = 0.0           # when the newest commit was sent
        self._said_down = False

    # --- called from the 20 ms loop (other thread) ---------------------------------------------------------------
    def ensure_started(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
            self._thread.start()

    def feed(self, pcm_bytes):
        if self.ok and self.loop is not None:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ("audio", pcm_bytes))

    def commit(self, cb=None):
        if self.ok and self.loop is not None:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ("commit", cb))
        elif cb is not None:
            cb(None)

    @property
    def pending(self):
        """A commit is in flight: partials now are about the audio before it, not the new burst."""
        if self._recv < self._sent and time.monotonic() - self._commit_t > COMMIT_TIMEOUT_S:
            self._drop_pending()                                # give up on lost commits
        return self._recv < self._sent or self._want is not None

    def _drop_pending(self):
        for n in range(self._recv + 1, self._sent + 1):
            for cb in self._cbs.pop(n, []):
                cb(None)
        self._recv = self._sent

    # --- the network thread ----------------------------------------------------------------------------------------
    async def _main(self):
        from websockets.asyncio.client import connect
        self.loop = asyncio.get_running_loop()
        key = _env("ELEVENLABS_API_KEY")
        if not key:
            loud_warning("NO ELEVENLABS_API_KEY in .env: loudness-only rule", ["my_system_eleven.py needs the key"])
            return
        while True:
            try:
                async with connect(URL, additional_headers={"xi-api-key": key}, max_size=2 ** 22, open_timeout=8) as ws:
                    first = json.loads(await asyncio.wait_for(ws.recv(), 8))
                    if first.get("message_type") != "session_started":
                        raise RuntimeError(f"refused: {str(first)[:150]}")
                    self.q = asyncio.Queue()
                    self._sent = self._recv = self._since = 0
                    self._cbs.clear()
                    self._want = None
                    self.ok = True
                    if self._said_down:
                        print("[eleven] connected again", flush=True)
                    else:
                        print("[eleven] scribe_v3_live connected", flush=True)
                    self._said_down = False
                    _log("[eleven] connected")
                    sender = asyncio.create_task(self._sender(ws))
                    try:
                        await self._reader(ws)
                    finally:
                        sender.cancel()
            except Exception as ex:
                line = f"[eleven] connection lost / failed: {ex!r}"[:220]
                print(line, flush=True)
                _log(line)
            self.ok = False
            for cbs in list(self._cbs.values()) + [self._want or []]:
                for cb in cbs:
                    cb(None)
            self._cbs.clear()
            self._want = None
            if not self._said_down:
                self._said_down = True
                loud_warning("ElevenLabs live is DOWN: temporary fallback to the loudness-only rule",
                             ["Reconnecting every 2 s; it switches back on its own."])
            await asyncio.sleep(2)

    async def _send(self, ws, data, commit):
        await ws.send(json.dumps({"message_type": "input_audio_chunk", "commit": commit, "sample_rate": M.SR,
                                  "audio_base_64": base64.b64encode(data).decode()}))

    async def _sender(self, ws):
        buf = []
        zero = np.zeros(M.FRAME_SAMPLES, dtype="<i2").tobytes()
        while True:
            kind, x = await self.q.get()
            if kind == "audio":
                buf.append(x)
                self._since += 1
            else:                                               # merge with any commit already waiting
                self._want = (self._want or []) + ([x] if x is not None else [])
            if self._want is not None and self._since >= MIN_COMMIT_FRAMES:
                data, buf = b"".join(buf) or zero, []
                self._sent += 1
                self._cbs[self._sent] = self._want
                self._want, self._since = None, 0
                self._commit_t = time.monotonic()
                await self._send(ws, data, True)
            elif len(buf) >= CHUNK_FRAMES:
                data, buf = b"".join(buf), []
                await self._send(ws, data, False)

    async def _reader(self, ws):
        async for raw in ws:
            m = json.loads(raw)
            kind = str(m.get("message_type", ""))
            if kind == "partial_transcript":
                tgt = self.target
                if tgt is not None and not self.pending:
                    try:
                        tgt._on_partial(m.get("text", ""))
                    except Exception as ex:
                        _log(f"[eleven] partial handler error {ex!r}")
            elif kind == "committed_transcript":
                if self._recv < self._sent:
                    self._recv += 1
                    for cb in self._cbs.pop(self._recv, []):
                        cb(m.get("text", ""))
            elif kind.startswith("committed_transcript"):
                pass                                              # the with-timestamps twin of the message above
            elif kind != "session_started":
                _log(f"[eleven] message {str(m)[:200]}")
                if "error" in kind or "quota" in kind or "limit" in kind:
                    print(f"[eleven] {str(m)[:200]}", flush=True)
                    self._drop_pending()                        # no answer will come for these commits


EL = Eleven()


class ElevenSystem(MySystem):
    def __init__(self):
        self._last_db = M.NO_PEAK
        self._bpk = M.NO_PEAK                    # loudest 20 ms of the open burst
        self._burst_epoch = -1                   # epoch in which the newest burst started
        self._spec_text = None
        super().__init__()
        EL.target = self                         # the newest harness session gets the transcripts

    # --- recogniser plumbing (replaces the Whisper worker) --------------------------------------------------------
    def _start_worker(self):
        EL.ensure_started()

    def _ensure_worker(self):
        EL.ensure_started()

    @property
    def _asr_live(self):
        return EL.ok

    def _log(self, line):
        _log(line)

    def _push(self, caller):
        super()._push(caller)
        self._last_db = loudness_db(caller)
        EL.feed(np.asarray(caller, dtype="<i2").tobytes())

    def on_scenario_start(self, context=None):
        super().on_scenario_start(context)
        EL.commit()

    def on_agent_start(self, text, duration_ms):
        super().on_agent_start(text, duration_ms)
        EL.commit()                              # the caller's earlier words are not part of the next burst

    def _track_burst(self, sound):
        prev = self._burst_id
        super()._track_burst(sound)
        if self._burst_id != prev:
            self._bpk = self._last_db
            self._burst_epoch = self._epoch
        elif self._in_burst and sound:
            self._bpk = max(self._bpk, self._last_db)

    def _request(self):
        """my_system asks for a decode as the burst grows (partials do that here) and once more when it ends: then we
        commit, so the next burst's partials start from nothing."""
        if not self._in_burst:
            EL.commit(self._on_final)            # short words («لحظة») often come only with the commit: judge it too

    def _on_final(self, text):
        if text:
            self._on_partial(text)

    def _request_eot(self):
        eot_id = self._eot_id
        t0 = time.perf_counter()

        def done(text):
            if text is None:
                return                           # no answer: _eot_ready takes the turn after 900 ms anyway
            inc, why = eot_incomplete(text)
            line = (f"[eot] ({(time.perf_counter() - t0) * 1000:.0f} ms): {text!r} -> "
                    f"{'WAIT' if inc else 'take turn'} ({why})")
            print(line, flush=True)
            self._log(line)
            with self._lock:
                if eot_id == self._eot_id:
                    self._eot_res = (eot_id, inc, why)
        EL.commit(done)

    # --- a partial transcript arrived (network thread) ----------------------------------------------------------
    def _on_partial(self, text):
        if not text or not text.strip():
            return
        with self._lock:
            epoch, bid, ref = self._epoch, self._burst_id, self._burst_ref
            if not self.agent_talking or self._burst_epoch != epoch:
                return                           # caller's own turn, or no caller sound during this utterance yet
            agent_text = " ".join(self.agent_hist)
            pk = self._bpk
        stop, why = decide(text, agent_text, peak_db=float(pk), ref_db=ref)
        spec = False
        if M.SPEC_STOP and not stop and ref is not None and pk - ref >= M.SPEC_REL_DB \
                and why == "acknowledgement or too little to tell":
            spec = len({t for t in tokens(text) if t not in NEUTRAL and not is_ack(t)}) >= 2
        with self._lock:
            if epoch != self._epoch:
                return
            if self._spec is not None and self.stopped:
                if stop:
                    self._spec = None            # confirmed
                elif not spec and text != self._spec_text:
                    self._resume_flag = True     # newer text does not back it up: take it back
                verdict = "confirm" if stop else "no change"
            elif self.stopped:
                return
            elif stop:
                self._stop_flag = True
                verdict = "STOP"
            elif spec:
                self._spec = (bid, self._n)
                self._spec_text = text
                self._stop_flag = True
                verdict = "SPEC-STOP"
            else:
                verdict = "keep"
        rel = f"{pk - ref:+.0f}" if ref is not None else "n/a"
        ms = (self._n - self._burst_start) * 1000 // M.SR
        line = f"[eleven] burst {bid} at {ms} ms (pk {pk:.0f} dB, vs caller {rel} dB): {text!r} -> {verdict} ({why})"
        print(line, flush=True)
        self._log(line)


if __name__ == "__main__":
    EL.ensure_started()
    for _ in range(100):                         # wait up to 10 s for the connection before serving
        if EL.ok:
            break
        time.sleep(0.1)
    if not EL.ok:
        loud_warning("ElevenLabs live NOT connected at start: loudness-only rule until it connects", [])
    run(ElevenSystem, TEAM, __file__)
