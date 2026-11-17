"""Make augmented copies of the dev calls: only the caller channel changes, so every label stays exact.

    python augment.py                       # caller volume -12, -6, +6 dB  ->  data/aug/vol-12, vol-6, vol+6
    python augment.py --gains -9 3

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", nargs="+", type=float, default=[-12, -6, 6])
    for g in ap.parse_args().gains:
        make_volume_set(g)


if __name__ == "__main__":
    main()
