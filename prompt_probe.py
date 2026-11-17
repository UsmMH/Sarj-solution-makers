"""Does an initial prompt stop Whisper from hearing «ترجمة نانسي قنقر» in short real speech? Offline, on all dev events.

For every labelled caller event, cut the caller channel like the live system does: pieces of 0.4 / 0.6 / 0.8 s from
just before the moment that matters (EVIDENT for STOP/EITHER events, ONSET for KEEP events). Transcribe each piece
with the live settings, with and without an initial prompt, and run decide() on the text (no loudness gate, so only
the words count).

    python prompt_probe.py

Report per variant: stock phrases on STOP pieces (bad: lost time), earliest piece that says STOP per STOP event
(good: sooner), and STOP decisions on KEEP pieces (bad: false stops).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from my_system import ASR
from ttbench.dataset import read_wav_stereo
from turn_rules import STOCK, decide, tokens

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = 16000
LENGTHS = (0.4, 0.6, 0.8)
# name -> (initial prompt, ms of audio before the moment, beam size for a retry when the fast decode is a stock phrase)
# Tested Nov 18: initial prompts are REJECTED: Whisper echoed them as speech («مكالمة هاتفية…» -> a false STOP on BC-D01,
# «ألو، إيه، تمام، طيب، لا، لحظة» on OV-D03) and did not reduce the stock phrase (6/27 either way).
VARIANTS = {
    "none": (None, 100, None),
    "pad400": (None, 400, None),
    "retry-beam5": (None, 100, 5),
    "pad400+retry": (None, 400, 5),
}


def pieces():
    for path in sorted(glob.glob("data/dev/labels/*.json")):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        x = read_wav_stereo(Path("data/dev/audio") / d["audio"]["file"])[:, 1].astype(np.float32) / 32768
        agent = " ".join(u["text"] for u in d["agent_utterances"])
        for ev in d["events"]:
            exp = ev["expected"]
            if exp == "TAKE_TURN":
                continue
            t0 = ev["evident_ms"] if exp in ("STOP", "EITHER") and ev.get("evident_ms") else ev["onset_ms"]
            for L in LENGTHS:
                yield ev["event_id"], exp, L, x, int(t0 * SR / 1000), agent


def _decode(audio, prompt, beam):
    segs, _ = ASR.model.transcribe(audio, language="ar", beam_size=beam, temperature=0.0,
                                   condition_on_previous_text=False, without_timestamps=True,
                                   max_new_tokens=24, initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segs)


def run(variant, items):
    prompt, pad_ms, retry_beam = variant
    out = []
    for eid, exp, L, x, at, agent in items:
        audio = x[max(0, at - pad_ms * SR // 1000): at + int(L * SR)]     # the piece ends L seconds after the moment
        text = _decode(audio, prompt, 1)
        if retry_beam and STOCK.search(" ".join(tokens(text))):
            text = _decode(audio, prompt, retry_beam)
        stop, why = decide(text, agent)
        out.append((eid, exp, L, text, stop, why, bool(STOCK.search(" ".join(tokens(text))))))
    return out


def main():
    ASR.load()
    items = list(pieces())
    print(f"{len(items)} pieces from {len({i[0] for i in items})} events\n")
    for name, variant in VARIANTS.items():
        res = run(variant, items)
        stop_p = [r for r in res if r[1] in ("STOP", "EITHER")]
        keep_p = [r for r in res if r[1] == "KEEP"]
        stock = sum(r[6] for r in stop_p)
        earliest = {}
        for eid, exp, L, text, stop, why, st in stop_p:
            if stop and (eid not in earliest or L < earliest[eid]):
                earliest[eid] = L
        n_stop_ev = len({r[0] for r in stop_p})
        false = [r for r in keep_p if r[4]]
        print(f"=== {name}: stock phrase on {stock}/{len(stop_p)} STOP pieces; STOP found for "
              f"{len(earliest)}/{n_stop_ev} stop events, by length {dict(sorted(map(lambda v: (v, list(earliest.values()).count(v)), set(earliest.values()))))}; "
              f"STOP on {len(false)}/{len(keep_p)} KEEP pieces")
        for eid, exp, L, text, stop, why, st in res:
            if eid.split(".")[0] in ("GI-D02", "AM-D02", "GI-D03", "OV-D03") or (exp == "KEEP" and stop):
                print(f"   {eid:10} {exp:6} {L:.1f}s {'STOP' if stop else 'keep':4} {text!r}  [{why}]")
        print()


if __name__ == "__main__":
    main()
