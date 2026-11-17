"""Would a stronger recogniser read the caller's first words? Local Whisper vs audio models on OpenRouter.

For every labelled caller segment in data/dev (the true text is in labels/*.json), cut the caller channel from
the segment start to EVIDENT + 600 ms (about what exists when a stop is due), transcribe it with each recogniser
and print it next to the true text. Not streaming: this measures reading quality, not latency.

    python asr_compare.py                                       # Whisper + default cloud models, STOP/EITHER events
    python asr_compare.py --models google/gemini-2.5-flash --keep   # also KEEP segments (backchannels, noise)
    python asr_compare.py --no-whisper
"""
import argparse
import base64
import glob
import http.client
import io
import json
import os
import sys
import time
import wave
from difflib import SequenceMatcher

import numpy as np

from llm_judge import _read_key
from turn_rules import normalize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = 16000
AFTER_EVIDENT_MS = 600
PROMPT = ("Transcribe this Arabic phone audio (Saudi dialect) exactly as spoken, in Arabic script. It may be cut off "
          "mid-word. Output only the transcript, nothing else. If there is no speech, output nothing.")
DEFAULT_MODELS = ["google/gemini-2.5-flash", "google/gemini-3.7-flash", "openai/gpt-audio-mini",
                  "mistralai/voxtral-small-24b-2507", "qwen/qwen3.8-omni-flash"]


def caller_channel(path):
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x[:, 1]                                                  # channels: [agent, user]


def clips(keep):
    out = []
    for path in sorted(glob.glob("data/dev/labels/*.json")):
        d = json.load(open(path, encoding="utf-8"))
        segs = {s["seg_id"]: s for s in d.get("user_segments", [])}
        audio = None
        for ev in d.get("events", []):
            if ev.get("expected") not in ("STOP", "EITHER") and not keep:
                continue
            if ev.get("expected") == "TAKE_TURN":
                continue
            for sid in ev.get("segment_ids", []):
                s = segs.get(sid)
                if not s or not s.get("text"):
                    continue
                if audio is None:
                    audio = caller_channel(os.path.join("data/dev/audio", d["audio"]["file"]))
                lo = max(0, s["start_ms"] - 100)
                hi = min(s["end_ms"], (ev.get("evident_ms") or s["start_ms"]) + AFTER_EVIDENT_MS)
                hi = max(hi, lo + 400)
                out.append((ev["event_id"], ev["expected"], s["text"], audio[lo * SR // 1000: hi * SR // 1000]))
    return out


def wav_b64(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(x.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def cloud(model, key, x):
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "input_audio", "input_audio": {"data": wav_b64(x), "format": "wav"}}]}]}
    t0 = time.perf_counter()
    try:
        c = http.client.HTTPSConnection("openrouter.ai", timeout=30)
        c.request("POST", "/api/v1/chat/completions", json.dumps(body).encode(),
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = c.getresponse()
        d = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        if r.status != 200:
            return f"<error {r.status}: {str(d.get('error', d))[:80]}>", ms
        return (d["choices"][0]["message"].get("content") or "").strip(), ms
    except Exception as ex:
        return f"<error {ex!r}>"[:100], (time.perf_counter() - t0) * 1000


def similarity(hyp, ref, n_chars):
    """How well the hypothesis matches the START of the true text (the clip only holds the start)."""
    h, r = normalize(hyp), normalize(ref)[: max(n_chars, 1)]
    return SequenceMatcher(None, h, r).ratio() if h and not hyp.startswith("<error") else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--keep", action="store_true", help="also KEEP segments (false-alarm check)")
    ap.add_argument("--no-whisper", action="store_true")
    args = ap.parse_args()
    items = clips(args.keep)
    key = _read_key()
    names = ([] if args.no_whisper else ["whisper-local"]) + args.models
    asr = None
    if not args.no_whisper:
        from my_system import ASR
        ASR.load()
        asr = ASR if ASR.ok else None
    totals = {n: [] for n in names}
    times = {n: [] for n in names}
    for eid, exp, ref, x in items:
        print(f"\n{eid} [{exp}] {len(x) / SR:.1f}s   TRUE: {ref}")
        n_chars = int(len(normalize(ref)) * min(1.0, len(x) / SR / 3.0)) + 4   # rough share of the text in the clip
        for n in names:
            if n == "whisper-local":
                if asr is None:
                    continue
                t0 = time.perf_counter()
                hyp = asr.transcribe(x.astype(np.float32) / 32768)[0]
                ms = (time.perf_counter() - t0) * 1000
            else:
                hyp, ms = cloud(n, key, x)
            s = similarity(hyp, ref, n_chars)
            totals[n].append(s)
            times[n].append(ms)
            print(f"   {n:34} {s:4.2f} {ms:5.0f} ms  {hyp}")
    print("\n=== mean match to the true start (0-1), median time per clip")
    for n in names:
        if totals[n]:
            print(f"   {n:34} {np.mean(totals[n]):.2f}   {np.median(times[n]):.0f} ms")


if __name__ == "__main__":
    main()
