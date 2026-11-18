"""Base Whisper turbo vs another faster-whisper model (e.g. the Arabic-dialect fine-tune), offline, same clips.

    python model_compare.py --other models/turbo-ar-dialectal-v2-ct2

1. dev pieces: 0.4 / 0.6 / 0.8 s from the moment that matters, with the live 400 ms lead-in (as prompt_probe.py)
2. the 9 hard dev caller clips with their true text (as asr_compare.py)
3. the team dataset (data/v3_bench): the same short pieces, plus whole caller clips vs label text
Reports per model: stock phrases on stop pieces, stops caught by piece length, STOP decisions on KEEP pieces,
match with the true text, and decode time. Nothing here changes the system.
"""
import argparse
import glob
import json
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from my_system import ASR_COMPUTE, add_cuda_dlls
from ttbench.dataset import read_wav_stereo
from turn_rules import STOCK, decide, normalize, tokens

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = 16000
LEAD_MS = 400
LENGTHS = (0.4, 0.6, 0.8)


def events(data_dir):
    """(event id, expected, caller audio, moment in samples, offset in samples, agent text, true caller text)"""
    for path in sorted(glob.glob(f"{data_dir}/labels/*.json")):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        x = read_wav_stereo(Path(data_dir) / "audio" / d["audio"]["file"])[:, 1].astype(np.float32) / 32768
        agent = " ".join(u["text"] for u in d.get("agent_utterances", []))
        segs = {s["seg_id"]: s for s in d.get("user_segments", [])}
        for ev in d["events"]:
            if ev.get("expected") not in ("STOP", "KEEP", "EITHER"):
                continue
            t0 = ev.get("evident_ms") if ev["expected"] != "KEEP" and ev.get("evident_ms") else ev["onset_ms"]
            ref = " ".join(segs[s].get("text", "") for s in ev.get("segment_ids", []) if s in segs)
            yield (ev["event_id"], ev["expected"], x, int(t0 * SR / 1000),
                   int((ev.get("offset_ms") or ev["onset_ms"] + 2000) * SR / 1000), agent, ref)


def run(model, data_dir, label):
    times, rows = [], []

    def dec(audio, n=24):
        t = time.perf_counter()
        segs, _ = model.transcribe(audio, language="ar", beam_size=1, temperature=0.0, condition_on_previous_text=False,
                                   without_timestamps=True, max_new_tokens=n)
        text = " ".join(s.text.strip() for s in segs)
        times.append((time.perf_counter() - t) * 1000)
        return text

    for eid, exp, x, at, off, agent, ref in events(data_dir):
        earliest = None
        stock = 0
        false = []
        for L in LENGTHS:
            text = dec(x[max(0, at - LEAD_MS * SR // 1000): at + int(L * SR)])
            stop, why = decide(text, agent)
            stock += bool(STOCK.search(" ".join(tokens(text))))
            if exp in ("STOP", "EITHER") and stop and earliest is None:
                earliest = L
            if exp == "KEEP" and stop:
                false.append((L, text, why))
        sim = None
        if ref:
            whole = dec(x[max(0, at - LEAD_MS * SR // 1000): off + 300 * SR // 1000], 64)
            sim = SequenceMatcher(None, normalize(ref), normalize(whole)).ratio()
        rows.append((eid, exp, earliest, stock, false, sim))
    stops = [r for r in rows if r[1] in ("STOP", "EITHER")]
    keeps = [r for r in rows if r[1] == "KEEP"]
    caught = {L: sum(1 for r in stops if r[2] is not None and r[2] <= L) for L in LENGTHS}
    sims = [r[5] for r in rows if r[5] is not None]
    print(f"  {label:10} stop events {len(stops)}: caught by 0.4s {caught[0.4]}, 0.6s {caught[0.6]}, 0.8s {caught[0.8]} | "
          f"stock phrase on {sum(r[3] for r in stops)}/{3 * len(stops)} stop pieces | "
          f"false STOP on {sum(len(r[4]) for r in keeps)}/{3 * len(keeps)} keep pieces | "
          f"text match {np.mean(sims):.2f} | decode median {statistics.median(times):.0f} ms")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True)
    a = ap.parse_args()
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    models = {"base turbo": WhisperModel("large-v3-turbo", device="cuda", compute_type=ASR_COMPUTE),
              "dialectal": WhisperModel(a.other, device="cuda", compute_type=ASR_COMPUTE)}
    for m in models.values():                              # warm-up
        m.transcribe(np.zeros(SR, dtype=np.float32), language="ar")
    for data_dir in ("data/dev", "data/v3_bench"):
        print(f"\n=== {data_dir}")
        res = {name: run(m, data_dir, name) for name, m in models.items()}
        a_rows, b_rows = res["base turbo"], res["dialectal"]
        for ra, rb in zip(a_rows, b_rows):                 # where the two disagree
            if ra[2] != rb[2] or bool(ra[4]) != bool(rb[4]):
                print(f"     {ra[0]:28} {ra[1]:6} first STOP piece: base {ra[2]} / dialectal {rb[2]}"
                      f"{'  | base false-stops' if ra[4] else ''}{'  | dialectal false-stops ' + repr(rb[4][0][1]) if rb[4] else ''}")


if __name__ == "__main__":
    main()
