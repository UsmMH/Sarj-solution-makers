"""B: speculative STOP on two loud content words, taken back by RESUME when the next decode does not confirm it.

    python -m pytest -q test_spec_resume.py
No GPU: decode results are handed to MySystem._apply() directly, as the worker thread would.
"""
import importlib
import os

import numpy as np
import pytest


@pytest.fixture()
def ms(monkeypatch):
    monkeypatch.setenv("SPEC_STOP", "on")
    monkeypatch.setenv("ASR", "off")                       # no worker thread, no model: we drive _apply ourselves
    import my_system
    m = importlib.reload(my_system)
    m.ASR.ok = True                                        # pretend Whisper is up so the ASR path runs
    yield m
    monkeypatch.delenv("SPEC_STOP")
    monkeypatch.delenv("ASR")
    importlib.reload(my_system)


def frames(s, n, level=0):
    out = []
    for _ in range(n):
        x = (np.random.default_rng(len(out)).normal(0, level, 320)).astype(np.int16) if level else np.zeros(320, np.int16)
        out.append(s.on_frame(np.zeros(320, np.int16), x))
    return [d for d in out if d]


def started(ms):
    s = ms.MySystem()
    s.on_scenario_start({})
    s.on_agent_start("نص الوكيل", 8000)
    frames(s, 50)
    return s


def test_unconfirmed_spec_stop_resumes(ms):
    s = started(ms)
    s._apply((s._epoch, False, True, 7))                   # spec STOP
    assert frames(s, 1) == ["STOP"]
    frames(s, 10)                                          # 200 ms
    s._apply((s._epoch, False, False, 7))                  # next decode: no evidence
    assert frames(s, 1) == ["RESUME"]
    assert frames(s, 20) == []


def test_confirmed_spec_stop_stays_stopped(ms):
    s = started(ms)
    s._apply((s._epoch, False, True, 7))
    assert frames(s, 1) == ["STOP"]
    s._apply((s._epoch, True, False, 7))                   # confirmed
    s._apply((s._epoch, False, False, 7))                  # a later non-stop decode must not undo a confirmed stop
    assert frames(s, 30) == []


def test_late_denial_does_not_resume(ms):
    s = started(ms)
    s._apply((s._epoch, False, True, 7))
    assert frames(s, 1) == ["STOP"]
    frames(s, 30)                                          # 600 ms > SPEC_RESUME_MS: too late for half credit
    s._apply((s._epoch, False, False, 7))
    assert frames(s, 20) == []


def test_spec_switch(monkeypatch):
    import my_system
    monkeypatch.setenv("SPEC_STOP", "off")
    assert importlib.reload(my_system).SPEC_STOP is False
    monkeypatch.delenv("SPEC_STOP")
    assert importlib.reload(my_system).SPEC_STOP is True        # on by default since Nov 20
