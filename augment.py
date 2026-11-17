"""Make augmented copies of the dev calls: only the caller channel changes, so every label stays exact.

    python augment.py                       # caller volume -12, -6, +6 dB  ->  data/aug/vol-12, vol-6, vol+6
    python augment.py --gains -9 3
    python augment.py --gains --stress      # only the stress sets: data/aug/loudstart+8, data/aug/click

Then play a set like the normal dev set (two windows, real time, as always):
    python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --data data/aug/vol-12 --name vol-12

The caller channel holds everything the caller's phone hears (their voice, room noise, TV), so turning it down
is a quieter line or a softer speaker; turning it up is a louder one. The agent channel is untouched.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ttbench.dataset import read_wav_stereo, write_wav_stereo

SRC = Path("data/dev")
OUT = Path("data/aug")


def make_volume_set(gain_db: float) -> Path:
    name = f"vol{gain_db:+g}".replace("+-", "-")
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "audio").mkdir(parents=True)
    shutil.copytree(SRC / "labels", dst / "labels")
    for extra in ("manifest.json", "events.csv"):
        if (SRC / extra).exists():
            shutil.copy2(SRC / extra, dst / extra)
    factor = 10 ** (gain_db / 20)
    clipped = 0
    for wav in sorted((SRC / "audio").glob("*.wav")):
        x = read_wav_stereo(wav).astype(np.float64)
        x[:, 1] *= factor                                         # column 1 = caller
        clipped += int((np.abs(x[:, 1]) > 32767).sum())
        write_wav_stereo(dst / "audio" / wav.name, np.clip(np.round(x), -32768, 32767).astype(np.int16))
    meta = {"source": str(SRC), "augmentation": "caller volume", "gain_db": gain_db, "clipped_samples": clipped}
    (dst / "augmentation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{dst}: caller {gain_db:+g} dB, {clipped} clipped samples")
    return dst


def _new_set(name: str) -> Path:
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "audio").mkdir(parents=True)
    shutil.copytree(SRC / "labels", dst / "labels")
    for extra in ("manifest.json", "events.csv"):
        if (SRC / extra).exists():
            shutil.copy2(SRC / extra, dst / extra)
    return dst


def _first_event_ms(label: dict):
    ons = [e["onset_ms"] for e in label.get("events", []) if e.get("expected") != "TAKE_TURN"]
    return min(ons) if ons else None


def make_loud_start_set(gain_db: float = 8.0) -> Path:
    """Stress test for the relative gate: the caller is louder at the start (shouted «ألو!») and normal later,
    so their early voice sets a high reference and a normal-level interruption looks 'quiet'."""
    dst = _new_set(f"loudstart{gain_db:+g}")
    factor, n_sets = 10 ** (gain_db / 20), 0
    for wav in sorted((SRC / "audio").glob("*.wav")):
        label = json.loads((SRC / "labels" / f"{wav.stem}.json").read_text(encoding="utf-8"))
        x = read_wav_stereo(wav).astype(np.float64)
        first = _first_event_ms(label)
        if first is not None:
            cut = max(0, first - 500) * 16                          # until 0.5 s before the first caller event
            x[:cut, 1] *= factor
            n_sets += 1
        write_wav_stereo(dst / "audio" / wav.name, np.clip(np.round(x), -32768, 32767).astype(np.int16))
    meta = {"source": str(SRC), "augmentation": "caller louder before the first event", "gain_db": gain_db,
            "calls_changed": n_sets}
    (dst / "augmentation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{dst}: caller {gain_db:+g} dB before the first event in {n_sets} calls")
    return dst


def make_click_set(at_ms: int = 100, length_ms: int = 40, level_db: float = -6.0) -> Path:
    """Stress test: one loud click on the caller line while the agent is still silent (a bump on the mic)."""
    dst = _new_set("click")
    rng = np.random.default_rng(0)
    amp = 32768 * 10 ** (level_db / 20)
    for wav in sorted((SRC / "audio").glob("*.wav")):
        x = read_wav_stereo(wav).astype(np.float64)
        a, b = at_ms * 16, (at_ms + length_ms) * 16
        x[a:b, 1] += amp * rng.uniform(-1, 1, b - a)                # white-noise click, ~level_db RMS-ish peak
        write_wav_stereo(dst / "audio" / wav.name, np.clip(np.round(x), -32768, 32767).astype(np.int16))
    meta = {"source": str(SRC), "augmentation": "one loud click on the caller line", "at_ms": at_ms,
            "length_ms": length_ms, "level_db": level_db}
    (dst / "augmentation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{dst}: {length_ms} ms click at {at_ms} ms, {level_db:g} dBFS")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", nargs="*", type=float, default=[-12, -6, 6], help="caller volume sets")
    ap.add_argument("--stress", action="store_true", help="also the loud-start and click stress sets")
    a = ap.parse_args()
    for g in a.gains:
        make_volume_set(g)
    if a.stress:
        make_loud_start_set()
        make_click_set()


if __name__ == "__main__":
    main()
