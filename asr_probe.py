"""ASR probe: can a local Whisper read what the dev callers say, and how fast? Read-only: changes nothing.

    python asr_probe.py small                            # quick first run (small download)
    python asr_probe.py large-v3-turbo                   # bigger model, compare accuracy and speed
    python asr_probe.py large-v3-turbo int8_float16      # if the GPU runs out of memory

For every labelled event in data/dev it decodes the caller channel twice:
  PRE  = only the first 600 ms of the burst (about what we would have when we must decide)
  FULL = the whole burst (capped at 3 s), an upper bound on what the model can hear
Results go to asr_probe_<model>_<device>.txt (UTF-8). Paste THAT file back, not the console text (Arabic scrambles there).

GPU on Windows needs the CUDA 12 libraries once:   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
"""
import glob
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows cache warning


def add_cuda_dlls():
    """Make the pip-installed CUDA 12 libraries findable (Windows). Returns the folders it added."""
    try:
        import nvidia
    except ImportError:
        return []
    added = []
    for base in list(nvidia.__path__):
        for sub in ("cublas", "cudnn"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)
    return added


ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data" / "dev"
SR = 16000
PAD_MS = 100          # start a little before the labelled onset
PREFIX_MS = 600       # "what we would have at decision time"
FULL_CAP_MS = 3000    # whole-call noise events are cut to 3 s


def load_caller(sid):
    with wave.open(str(DEV / "audio" / f"{sid}.wav")) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    return d[:, 1].astype(np.float32) / 32768.0


def event_windows(label):
    """(event_id, expected, what_was_said, onset_ms, offset_ms) for every event that has both times."""
    texts = {s["seg_id"]: s.get("text") or f"[{s.get('kind', '?')}]" for s in label["user_segments"]}
    out = []
    for e in label["events"]:
        if "onset_ms" not in e or "offset_ms" not in e:
            continue
        said = " ".join(texts.get(x, "") for x in e.get("segment_ids", []))
        out.append((e["event_id"], e["expected"], said, e["onset_ms"], e["offset_ms"]))
    return out


def decode(model, audio):
    t = time.perf_counter()
    segs, _ = model.transcribe(audio, language="ar", beam_size=1, temperature=0.0,
                               condition_on_previous_text=False, without_timestamps=True,
                               max_new_tokens=24)    # cap: stops the "هههههه..." repetition loops
    segs = list(segs)
    ms = (time.perf_counter() - t) * 1000
    text = " ".join(s.text.strip() for s in segs)
    nsp = max((s.no_speech_prob for s in segs), default=1.0)
    return text, nsp, ms


def load(name, compute):
    from faster_whisper import WhisperModel
    silence = np.zeros(SR, dtype=np.float32)
    try:
        model = WhisperModel(name, device="cuda", compute_type=compute)
        decode(model, silence)      # warm-up: loads the GPU kernels, and shows up a missing CUDA library
        return model, f"cuda/{compute}"
    except Exception as ex:         # a hard crash instead of an exception is possible too: paste what you see
        print("CUDA failed:", repr(ex)[:400])
        print("Falling back to CPU int8 ...")
        model = WhisperModel(name, device="cpu", compute_type="int8")
        decode(model, silence)
        return model, "cpu/int8"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    compute = sys.argv[2] if len(sys.argv) > 2 else "float16"
    dirs = add_cuda_dlls()
    print("CUDA library folders found:", dirs if dirs else "none (GPU will fail until the pip libraries are installed)")
    t0 = time.perf_counter()
    model, where = load(name, compute)
    lines = [f"model={name}  device={where}  load+warmup={time.perf_counter() - t0:.1f}s", "",
             "event | expected | said | PRE 600 ms: heard (no_speech_prob, ms) | FULL: heard (no_speech_prob, ms)"]
    times = []
    for lp in sorted(glob.glob(str(DEV / "labels" / "*.json"))):
        with open(lp, encoding="utf-8") as f:
            label = json.load(f)
        audio = load_caller(label["scenario_id"])
        for eid, exp, said, a, b in event_windows(label):
            s = max(0, int((a - PAD_MS) * SR / 1000))
            e_full = int(min(b, a + FULL_CAP_MS) * SR / 1000)
            e_pre = int(min(b, a + PREFIX_MS) * SR / 1000)
            t1, n1, m1 = decode(model, audio[s:e_pre])
            if e_pre >= e_full:
                t2, n2, m2 = t1, n1, m1
            else:
                t2, n2, m2 = decode(model, audio[s:e_full])
                times.append(m2)
            times.append(m1)
            lines.append(f"{eid} | {exp} | {said} | PRE: {t1 or '-'} ({n1:.2f}, {m1:.0f} ms)"
                         f" | FULL: {t2 or '-'} ({n2:.2f}, {m2:.0f} ms)")
    lines += ["", f"decode time per clip: median {statistics.median(times):.0f} ms, max {max(times):.0f} ms"]
    out = ROOT / f"asr_probe_{name.replace('/', '_')}_{where.replace('/', '-')}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(lines[0])
    print(lines[-1])
    print(f"Wrote {out.name}: paste its contents back.")


if __name__ == "__main__":
    main()