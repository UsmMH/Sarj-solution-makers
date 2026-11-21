"""SUBMITTED SYSTEM (Nov 21 morning): the hybrid (my_system_hybrid.py) with a narrower rescue. Pre-registered in
analysis/PREREG_cue_rescue_nov21.md before any run; passed (check fold 95.1 vs hybrid 94.0/94.7, same false stops).

ElevenLabs may rescue a burst (same Whisper-veto condition as my_system_hybrid.py) only when its text has a stop CUE or
PHRASE; never on "3+ words" or "question" (fluent sentences are where background TV lives: the one TV rescue seen was
«في الشأن المحلي، أعلنت»). A bare «لا» cue is only judged on ElevenLabs' final text of the burst and never when the
rest is a polite ending («لا لا إن شاء الله» = "not at all": KEEP, the mentors' example).

    python my_system_hybrid_cue.py            # the final run (add --host 0.0.0.0); needs the venv, internet and ELEVENLABS_API_KEY in .env
"""
import re

import my_system as M
import my_system_hybrid as H
from my_system_hybrid import *  # noqa: F401,F403
from turn_rules import (NEUTRAL, PHRASE_RE, STOP_CUES, decide as rules_decide, find_cue, is_ack, normalize,
                        tokens)

TEAM = "Team 4"

POLITE_PHRASES = [normalize(p) for p in (
    "ان شاء الله|انشاء الله|ان شالله|انشالله|انشاءالله|الله يعافيك|الله يعافيكم|الله يسلمك|الله يسلمكم|الله يخليك|الله يحفظك|"
    "الله يبارك فيك|الله يجزاك خير|جزاك الله خير|تسلم|تسلمين|تسلمون|ما عليك|ماعليك|ما عليكم|ولا يهمك|ولا يهمكم|عادي|لا باس|"
    "ما فيه مشكله|ما في مشكله|مافي مشكله").split("|")]      # same list as turn_rules_next.py (Nov 19); no «شكرا»:
                                                             # «لا شكرا» during an offer is a refusal, a floor bid
POLITE_RE = re.compile(r"(?:^| )(" + "|".join(re.escape(p) for p in sorted(POLITE_PHRASES, key=len, reverse=True)) + r")(?: |$)")


def polite_no(text: str) -> bool:
    """«لا» + only a polite ending («لا لا إن شاء الله», «لا الله يعافيك»): not a floor bid. Another cue, a phrase bid,
    or two more real words keep it a stop."""
    toks = tokens(text)
    norm = " ".join(toks)
    if not POLITE_RE.search(norm):
        return False
    if any(t in STOP_CUES and t != "لا" for t in toks) or PHRASE_RE.search(norm):
        return False
    left = [t for t in POLITE_RE.sub(" ", norm).split() if t != "لا" and t not in NEUTRAL and not is_ack(t)]
    return len(left) < 2


class MySystem(H.MySystem):
    def _el_final(self, text):
        if text:
            self._judge(text, final=True)

    def _on_partial(self, text):
        self._judge(text, final=False)

    def _judge(self, text, final):
        if not text or not text.strip():
            return
        with self._lock:
            epoch, bid, ref = self._epoch, self._burst_id, self._burst_ref
            if not self.agent_talking or self.stopped or self._stop_flag or self._burst_epoch != epoch:
                return
            w_why = self._w_why.get((epoch, bid))
            agent_text = " ".join(self.agent_hist)
            pk = self._bpk
        if w_why is not None and w_why not in H.RESCUE_OK:
            return
        stop, why = rules_decide(text, agent_text, peak_db=float(pk), ref_db=ref)
        if not stop or not (why.startswith("cue") or why.startswith("phrase")):
            return                                   # cue-only rescue: no "3+ words" / question from ElevenLabs
        if why == "cue «لا»":
            if not final or polite_no(text):
                return                               # bare «لا»: wait for the whole burst; polite «لا» never stops
        with self._lock:
            if epoch == self._epoch and self.agent_talking and not self.stopped:
                self._stop_flag = True
        rel = f"{pk - ref:+.0f}" if ref is not None else "n/a"
        line = (f"[rescue-cue] burst {bid} (pk {pk:.0f} dB, vs caller {rel} dB, whisper: {w_why}, "
                f"{'final' if final else 'partial'}): {text!r} -> STOP ({why})")
        print(line, flush=True)
        self._log(line)


if __name__ == "__main__":
    M.ASR.load()
    import time
    H.EL.ensure_started()
    for _ in range(100):
        if H.EL.ok:
            break
        time.sleep(0.1)
    if not H.EL.ok:
        M.loud_warning("ElevenLabs NOT connected: running as the plain Whisper system until it connects", [])
    M.run(MySystem, TEAM, __file__)
