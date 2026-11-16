"""Step 2a: adaptive noise floor + burst-peak gate + short-gap tolerance on the STOP path.

Changes vs step 1 (everything else, including the TAKE_TURN path, is unchanged):
  * PEAK_DB: a burst of sound only becomes a STOP if its loudest 20 ms frame reaches this level.
    In the dev events, genuine interruptions peak at >= -16.7 dB; most backchannels/noise peak lower.
    (Intensity cue only: treat as a stopgap, NOT as ground truth for the hidden set. See notes.)
  * GAP_TOLERANCE_FRAMES: a single quiet frame no longer resets the burst, so short words in loud
    noise (GI-D03 "لحظة", 450 ms) can still accumulate STOP_AFTER_MS.

    python -m ttbench.play --fast --calls all     # quick check, ~2.5 min
    python -m ttbench.play --calls all            # real-time run, ~10 min
"""
import collections
import os

import numpy as np

from ttbench.server import System, run

TEAM = "Team 4 (for now :D)"            # your team name, shown on the scoreboard

FRAME_MS = 20               # each on_frame call covers 20 ms of audio
LOUD_DB = -42               # absolute minimum for "sound" (the old fixed threshold)
STOP_AFTER_MS = 200         # sound for this long while the agent talks  -> STOP candidate
TAKE_TURN_AFTER_MS = 700    # quiet for this long after the caller spoke -> TAKE_TURN (unchanged baseline)

# --- Step 1 knobs ------------------------------------------------------------------------------------
FLOOR_WINDOW_MS = 3000      # how much recent caller audio defines "the noise floor"
FLOOR_PERCENTILE = 20       # low percentile of that window = the floor
MARGIN_DB = 10              # caller must be this far above the floor to count as sound

# --- Step 2a knobs -----------------------------------------------------------------------------------
PEAK_DB = -20.0             # burst must reach this peak (dB) before it can trigger STOP
GAP_TOLERANCE_FRAMES = 2    # this many CONSECUTIVE quiet frames end a burst (2 = 40 ms)

NO_PEAK = -120.0


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
        self.gap = 0                    # consecutive quiet frames inside the current burst
        self.burst_peak = NO_PEAK       # loudest frame of the current burst
        self.quiet_ms = 0
        self.caller_spoke = False
        self.hist = collections.deque(maxlen=FLOOR_WINDOW_MS // FRAME_MS)   # recent caller levels (dB)

    # --- what is happening in the call -------------------------------------------------------------------
    def on_scenario_start(self, *args, **kwargs):
        # New call = possibly new noise. Forget the old floor. (Harmless if the harness never calls this.)
        self.hist.clear()

    def on_agent_start(self, text, duration_ms):
        self.agent_talking, self.agent_text, self.stopped = True, text, False
        self.sound_ms, self.gap, self.burst_peak = 0, 0, NO_PEAK

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
            if loud_stop:
                self.sound_ms += FRAME_MS
                self.gap = 0
                self.burst_peak = max(self.burst_peak, db)
            else:
                self.gap += 1
                if self.gap >= GAP_TOLERANCE_FRAMES:        # burst really ended
                    self.sound_ms, self.burst_peak = 0, NO_PEAK

            if (self.sound_ms >= STOP_AFTER_MS and self.burst_peak >= PEAK_DB
                    and not self.stopped):
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