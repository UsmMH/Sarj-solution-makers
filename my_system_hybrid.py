"""SUBMITTED SYSTEM (Nov 21): Whisper (my_system.py, unchanged) + ElevenLabs live as a "rescue" reader. Pre-registered
in analysis/PREREG_hybrid_nov20.md before any run; passed its rule live (check fold 92.7 -> 94.0, no new false stop).

An ElevenLabs transcript of the open caller burst may trigger STOP only when decide() on it says STOP (all vetoes and
loudness gates apply, with the same burst peak and caller reference voice) AND Whisper has not actively vetoed the burst
(its latest decode is missing, empty, a stock phrase, or "acknowledgement or too little to tell"). ElevenLabs gets no
speculative or default stop, and no say in the end of turn. ElevenLabs down -> exactly my_system.py.

    python my_system_hybrid.py            # the final run (add --host 0.0.0.0); needs the venv (Whisper on the GPU), internet and ELEVENLABS_API_KEY in .env
"""
import my_system as M
from my_system import *  # noqa: F401,F403
from my_system_eleven import EL
from turn_rules import decide as rules_decide

TEAM = "Team 4"

RESCUE_OK = {"nothing heard", "stock phrase the ASR invents on noise", "acknowledgement or too little to tell"}
_last = {"why": None}
_decide = M.decide


def _decide_spy(text, agent_text="", peak_db=None, ref_db=None):
    r = _decide(text, agent_text, peak_db=peak_db, ref_db=ref_db)
    _last["why"] = r[1]
    return r


M.decide = _decide_spy          # my_system._decode looks `decide` up in its module: we only record Whisper's reason


class MySystem(M.MySystem):
    def __init__(self):
        self._w_why = {}                         # (epoch, burst_id) -> Whisper's latest reason for that burst
        self._bpk = M.NO_PEAK
        self._last_db = M.NO_PEAK
        self._burst_epoch = -1
        super().__init__()
        EL.target = self
        EL.ensure_started()

    # --- feed ElevenLabs the same caller audio -----------------------------------------------------------------
    def _push(self, caller):
        super()._push(caller)
        self._last_db = M.loudness_db(caller)
        EL.feed(caller.astype("<i2").tobytes())

    def on_scenario_start(self, context=None):
        super().on_scenario_start(context)
        self._w_why.clear()
        EL.commit()

    def on_agent_start(self, text, duration_ms):
        super().on_agent_start(text, duration_ms)
        EL.commit()                              # the next burst starts with fresh text

    def _track_burst(self, sound):
        prev = self._burst_id
        super()._track_burst(sound)
        if self._burst_id != prev:
            self._bpk = self._last_db
            self._burst_epoch = self._epoch
        elif self._in_burst and sound:
            self._bpk = max(self._bpk, self._last_db)

    def _request(self):
        super()._request()
        if not self._in_burst:                   # burst over: ElevenLabs' final text of it is judged too
            EL.commit(self._el_final)

    # --- Whisper's verdict per burst -------------------------------------------------------------------------
    def _decode(self, req):
        res = super()._decode(req)
        if res is not None:
            with self._lock:
                self._w_why[(req[0], req[1])] = _last["why"]
        return res

    # --- ElevenLabs text (network thread) -----------------------------------------------------------------------
    def _el_final(self, text):
        if text:
            self._on_partial(text)

    def _on_partial(self, text):
        if not text or not text.strip():
            return
        with self._lock:
            epoch, bid, ref = self._epoch, self._burst_id, self._burst_ref
            if not self.agent_talking or self.stopped or self._stop_flag or self._burst_epoch != epoch:
                return
            w_why = self._w_why.get((epoch, bid))
            agent_text = " ".join(self.agent_hist)
            pk = self._bpk
        if w_why is not None and w_why not in RESCUE_OK:
            return                               # Whisper vetoed this burst (echo, side talk, quiet, ...)
        stop, why = rules_decide(text, agent_text, peak_db=float(pk), ref_db=ref)
        if not stop:
            return
        with self._lock:
            if epoch == self._epoch and self.agent_talking and not self.stopped:
                self._stop_flag = True
        rel = f"{pk - ref:+.0f}" if ref is not None else "n/a"
        line = (f"[rescue] burst {bid} (pk {pk:.0f} dB, vs caller {rel} dB, whisper: {w_why}): {text!r} -> STOP ({why})")
        print(line, flush=True)
        self._log(line)


if __name__ == "__main__":
    M.ASR.load()
    import time
    EL.ensure_started()
    for _ in range(100):
        if EL.ok:
            break
        time.sleep(0.1)
    if not EL.ok:
        M.loud_warning("ElevenLabs NOT connected: running as the plain Whisper system until it connects", [])
    M.run(MySystem, TEAM, __file__)
