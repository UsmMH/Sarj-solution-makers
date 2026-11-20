"""TRIAL, not submitted: "default stop". The Whisper system (my_system.py) plus one rule:

    a caller who keeps talking for DS_MIN_MS at about their own loudness, with at least one word that is not a
    backchannel, is taking the floor -> STOP, even without a cue word, a question or 3 real words.

Idea from the practice-checkpoint board: the top teams had our Restraint (S 0.91-0.96) but far higher Responsiveness
(R 0.87-0.97 vs 0.63); our misses were overlaps and dialect, where the words did not prove a floor bid. Here the words
only have to fail to prove a backchannel / echo / side talk / TV. Everything decide() vetoes still vetoes.
Numbers from principles, not from dev scores: backchannels («إيه», «أها», «ممم») are short (well under 1 s); the
loudness line is the existing quiet gate (WEAK_EVIDENCE_REL_DB = -6 dB).

    python sim.py --data data/dev --name sim-ds__dev --system my_system_ds
    python my_system_ds.py          # live, like my_system.py
"""
import my_system as M
from my_system import *  # noqa: F401,F403  (ASR, TEAM, ... for sim.py and run())
from turn_rules import NEUTRAL, is_ack, tokens

DS_MIN_MS = 1000
DS_REL_DB = -6.0
_now = {"burst_ms": 0}
_decide = M.decide


def decide(text, agent_text="", peak_db=None, ref_db=None):
    stop, why = _decide(text, agent_text, peak_db=peak_db, ref_db=ref_db)
    if (not stop and why == "acknowledgement or too little to tell" and ref_db is not None and peak_db is not None
            and peak_db - ref_db >= DS_REL_DB and _now["burst_ms"] >= DS_MIN_MS
            and any(t not in NEUTRAL and not is_ack(t) for t in tokens(text))):
        return True, f"default stop: {_now['burst_ms']} ms of speech at the caller's level"
    return stop, why


M.decide = decide          # my_system._decode looks `decide` up in its module at call time


class MySystem(M.MySystem):
    def _decode(self, req):
        _now["burst_ms"] = (req[3] - req[2]) * 1000 // M.SR      # end - start of the burst, lead-in not counted
        return super()._decode(req)


if __name__ == "__main__":
    M.ASR.load()
    M.run(MySystem, M.TEAM, __file__)
