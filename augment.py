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


# --- robustness sets (Nov 20): conditions the hidden calls may have and the 24 dev calls do not. Every parameter is a
# textbook value (telephone band, G.711, RT60, SNR), chosen before any run, never tuned on a score.
SR = 16000


def _speech_rms(x: np.ndarray) -> float:
    """Loud-speech level of a channel: the 95th percentile of 20 ms frame RMS (silence does not dilute it)."""
    n = len(x) // 320 * 320
    fr = np.sqrt((x[:n].reshape(-1, 320) ** 2).mean(axis=1))
    return float(np.percentile(fr, 95)) if len(fr) else 1.0


def _colored_noise(n: int, power: float, rng) -> np.ndarray:
    """power 1 = pink (1/f), 2 = brown (1/f^2, rumble like a car or a fan). Unit RMS."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    y = np.fft.irfft(spec / f ** (power / 2), n)
    return y / (np.sqrt((y ** 2).mean()) + 1e-12)


def _phone(x: np.ndarray) -> np.ndarray:
    """Narrow-band telephone: 300-3400 Hz band, then G.711 mu-law 8-bit companding (the classic phone codec)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(spec, len(x)) / 32768
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.minimum(np.abs(y), 1)) / np.log1p(mu)
    c = np.round(c * 127) / 127                                     # 8 bits
    return np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu * 32768


def _reverb(x: np.ndarray, rt60: float, rng) -> np.ndarray:
    """Room echo: synthetic impulse response (direct sound + exponentially decaying noise tail), same loudness after."""
    t = np.arange(int(rt60 * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * 0.3
    rir[0] = 1.0
    n = len(x) + len(rir) - 1
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(rir, n), n)[: len(x)]
    return y * (_speech_rms(x) / (_speech_rms(y) + 1e-12))


def make_caller_set(name: str, fn, meta: dict) -> Path:
    dst = _new_set(name)
    rng = np.random.default_rng(0)
    for wav in sorted((SRC / "audio").glob("*.wav")):
        x = read_wav_stereo(wav).astype(np.float64)
        x[:, 1] = fn(x[:, 1], rng)
        write_wav_stereo(dst / "audio" / wav.name, np.clip(np.round(x), -32768, 32767).astype(np.int16))
    (dst / "augmentation.json").write_text(json.dumps({"source": str(SRC), **meta}, indent=2), encoding="utf-8")
    print(f"{dst}: {meta['augmentation']}")
    return dst


def make_robust_sets():
    make_caller_set("phone", lambda x, rng: _phone(x),
                    {"augmentation": "telephone band 300-3400 Hz + G.711 mu-law 8-bit"})
    make_caller_set("noise-pink15", lambda x, rng: x + _colored_noise(len(x), 1, rng) * _speech_rms(x) / 10 ** (15 / 20),
                    {"augmentation": "steady pink noise, 15 dB below the caller's loud speech", "snr_db": 15})
    make_caller_set("noise-brown5", lambda x, rng: x + _colored_noise(len(x), 2, rng) * _speech_rms(x) / 10 ** (5 / 20),
                    {"augmentation": "steady brown (car / fan rumble) noise, 5 dB below the caller's loud speech",
                     "snr_db": 5})
    make_caller_set("reverb", lambda x, rng: _reverb(x, 0.5, rng),
                    {"augmentation": "room echo, synthetic impulse response, RT60 0.5 s", "rt60_s": 0.5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", nargs="*", type=float, default=[-12, -6, 6], help="caller volume sets")
    ap.add_argument("--stress", action="store_true", help="also the loud-start and click stress sets")
    ap.add_argument("--robust", action="store_true", help="also phone, pink / brown noise and reverb sets")
    a = ap.parse_args()
    for g in a.gains:
        make_volume_set(g)
    if a.stress:
        make_loud_start_set()
        make_click_set()
    if a.robust:
        make_robust_sets()


if __name__ == "__main__":
    main()
