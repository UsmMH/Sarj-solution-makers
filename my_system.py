"""Step 1: adaptive noise floor on the STOP path.

Everything is the loudness baseline EXCEPT how "sound" is decided while the agent is talking:
instead of a fixed -42 dB, the caller must be MARGIN_DB above the recent noise floor (and never
below the old -42 dB). The TAKE_TURN (end-of-turn) path is deliberately left exactly as in the
baseline so its score (29) should not move; only R/S should.

    python -m ttbench.play --fast --calls all     # quick check, ~2.5 min
    python -m ttbench.play --calls all            # real-time run, ~10 min
"""
import collections

import numpy as np

from ttbench.server import System, run

TEAM = "my-team"            # your team name, shown on the scoreboard

FRAME_MS = 20               # each on_frame call covers 20 ms of audio
LOUD_DB = -42               # absolute minimum for "sound" (the old fixed threshold)
STOP_AFTER_MS = 200         # sound for this long while the agent talks  -> STOP
TAKE_TURN_AFTER_MS = 700    # quiet for this long after the caller spoke -> TAKE_TURN (unchanged baseline)

# --- Step 1 knobs (tune these one at a time) ---------------------------------------------------------
FLOOR_WINDOW_MS = 3000      # how much recent caller audio defines "the noise floor"
FLOOR_PERCENTILE = 20       # low percentile of that window = the floor (speech bursts don't lift it much)
MARGIN_DB = 10              # caller must be this far above the floor to count as sound


def loudness_db(samples: np.ndarray) -> float:
    """Loudness of 20 ms of 16-bit audio, in dB below full scale (silence is about -90, speech -40 to -15)."""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768
    return 20 * np.log10(rms + 1e-12)


class MySystem(System):
    def __init__(self):
        self.agent_talking = False
        self.agent_text = ""
        self.stopped = False
        self.sound_ms = 0
        self.quiet_ms = 0
        self.caller_spoke = False
        self.hist = collections.deque(maxlen=FLOOR_WINDOW_MS // FRAME_MS)   # recent caller levels (dB)

    # --- what is happening in the call -------------------------------------------------------------------
    def on_scenario_start(self, *args, **kwargs):
        # New call = possibly new noise. Forget the old floor. (Harmless if the harness never calls this.)
        self.hist.clear()

    def on_agent_start(self, text, duration_ms):
        self.agent_talking, self.agent_text, self.stopped, self.sound_ms = True, text, False, 0

    def on_agent_end(self):
        self.agent_talking, self.caller_spoke, self.quiet_ms = False, False, 0

    # --- every 20 ms: decide ---------------------------------------------------------------------------
    def on_frame(self, agent, caller):
        db = loudness_db(caller)

        # The floor is tracked on EVERY frame, talking or not, so it is ready when the agent starts.
        self.hist.append(db)
        floor = float(np.percentile(self.hist, FLOOR_PERCENTILE))

        loud_stop = db > max(LOUD_DB, floor + MARGIN_DB)    # adaptive: used only for STOP
        loud_eot = db > LOUD_DB                              # unchanged baseline: used only for TAKE_TURN

        if self.agent_talking:
            self.sound_ms = self.sound_ms + FRAME_MS if loud_stop else 0
            if self.sound_ms >= STOP_AFTER_MS and not self.stopped:
                self.stopped = True
                return "STOP"
            return None

        if loud_eot:
            self.caller_spoke, self.quiet_ms = True, 0
        elif self.caller_spoke:
            self.quiet_ms += FRAME_MS
            if self.quiet_ms >= TAKE_TURN_AFTER_MS:
                self.caller_spoke = False
                return "TAKE_TURN"
        return None


if __name__ == "__main__":
    run(MySystem, TEAM, __file__)