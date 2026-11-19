"""Is this burst the caller's own voice? A speaker-embedding check, used only to VETO a STOP (never to create one).

Model: WeSpeaker ResNet34-LM trained on VoxCeleb (CC BY 4.0), `voxceleb_resnet34_LM.onnx` (26.5 MB), run on the CPU with
onnxruntime so it never competes with Whisper for the GPU. Input = 80-dim Kaldi-style log-mel features (25 ms / 10 ms),
output = a 256-number voice fingerprint; two fingerprints are compared with the cosine.

Idea: while the agent is quiet the caller answers, so we hear the caller alone. Those clean stretches give the caller's
fingerprint. A burst during the agent's speech that is clearly a different voice (a teammate, a TV presenter, the room)
is not the caller asking for the floor, whatever its words say.
"""
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "speaker-wespeaker-r34lm" / "voxceleb_resnet34_LM.onnx"
SR = 16000
_N_FFT, _WIN, _HOP, _MELS = 512, 400, 160, 80
_MEL_BANK = None
_HAMMING = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(_WIN) / (_WIN - 1))).astype(np.float32)


def _mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def _bank():
    """Kaldi's triangular mel filters (20 Hz to Nyquist) over the 257 FFT bins."""
    global _MEL_BANK
    if _MEL_BANK is None:
        lo, hi = _mel(20.0), _mel(SR / 2)
        pts = lo + (hi - lo) * np.arange(_MELS + 2) / (_MELS + 1)
        bins = _mel(np.arange(_N_FFT // 2 + 1) * SR / _N_FFT)
        bank = np.zeros((_MELS, _N_FFT // 2 + 1), dtype=np.float32)
        for m in range(_MELS):
            left, centre, right = pts[m], pts[m + 1], pts[m + 2]
            up = (bins - left) / (centre - left)
            down = (right - bins) / (right - centre)
            bank[m] = np.maximum(0.0, np.minimum(up, down))
        _MEL_BANK = bank
    return _MEL_BANK


def fbank(audio: np.ndarray) -> np.ndarray:
    """audio: float32 in [-1, 1] at 16 kHz -> (frames, 80) log-mel, mean-normalised, as WeSpeaker expects."""
    x = audio.astype(np.float32) * 32768.0
    if len(x) < _WIN:
        x = np.pad(x, (0, _WIN - len(x)))
    n = 1 + (len(x) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)                              # remove DC per frame
    fr = np.concatenate([fr[:, :1] * (1 - 0.97), fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1)   # pre-emphasis
    fr = fr * _HAMMING
    spec = np.abs(np.fft.rfft(fr, n=_N_FFT, axis=1)) ** 2
    mel = np.log(np.maximum(spec @ _bank().T, np.finfo(np.float32).eps))
    return (mel - mel.mean(axis=0, keepdims=True)).astype(np.float32)


class SpeakerGate:
    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2                                       # leave the CPU to the audio loop
        self.sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])

    def embed(self, audio: np.ndarray) -> np.ndarray:
        feats = fbank(audio)[None]
        e = self.sess.run(None, {"feats": feats})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)

    @staticmethod
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
