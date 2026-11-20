"""Default stop (Nov 21): >= 1 s of speech at the caller's own loudness (>= -3 dB vs their reference voice) with at least
one non-backchannel word -> STOP, even without a cue, a question or 3 words. Everything decide() vetoes still vetoes.

    python -m pytest -q test_default_stop.py
No GPU: Whisper is replaced by a stub that returns a fixed text; _decode() runs as the worker thread would run it.
"""
import importlib

import numpy as np
import pytest

SR = 16000


@pytest.fixture()
def ms(monkeypatch):
    monkeypatch.setenv("ASR", "off")                       # no worker thread, no model
    import my_system
    m = importlib.reload(my_system)
    m.ASR.ok = True
    yield m
    monkeypatch.delenv("ASR")
    importlib.reload(my_system)


def decode(ms, text, burst_s, rel_db=0.0, monkeypatch=None):
    """Push a steady tone burst of burst_s seconds, decode it as `text`; the caller's reference voice is pk - rel_db."""
    s = ms.MySystem()
    s.on_scenario_start({})
    s.on_agent_start("موعدك يوم الأحد الساعة عشرة", 8000)
    amp = 0.1 * 32768
    n = int((burst_s + 0.5) * SR)
    tone = (amp * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.int16)
    s._push(tone)
    pk = 20 * np.log10(0.1 / np.sqrt(2))
    ms.ASR.transcribe = lambda audio, max_new_tokens=24: (setattr(ms.ASR, "last_ms", 280.0), (text, -0.2))[1]
    start = s._n - int(burst_s * SR)
    return s._decode((s._epoch, 1, start, s._n, s.agent_text, pk - rel_db))


def test_one_word_for_a_second_at_caller_level_stops(ms):
    _, stop, spec, _ = decode(ms, "الأحد", 1.2)
    assert stop and not spec


def test_short_burst_does_not_default_stop(ms):
    _, stop, _, _ = decode(ms, "الأحد", 0.6)
    assert not stop


def test_quieter_than_the_caller_does_not_default_stop(ms):
    _, stop, _, _ = decode(ms, "الأحد", 1.2, rel_db=-5.0)    # 5 dB under the caller: TV / someone else / echo
    assert not stop


def test_backchannels_never_default_stop(ms):
    _, stop, _, _ = decode(ms, "إيه إيه", 1.5)
    assert not stop


def test_stock_phrase_never_default_stops(ms):
    _, stop, _, _ = decode(ms, "ترجمة نانسي قنقر", 1.5)
    assert not stop


def test_switch_off(ms, monkeypatch):
    monkeypatch.setattr(ms, "DEFAULT_STOP", False)
    _, stop, _, _ = decode(ms, "الأحد", 1.2)
    assert not stop
