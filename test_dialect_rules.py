"""Unit tests for the dialect additions to turn_rules.py (Nov 20). Written BEFORE the rules were added and before any
benchmark run, from phrases composed for this file (not taken from dev / train / validation transcripts).

    python -m pytest -q test_dialect_rules.py

Each phrase is tagged with the region it is typical of. Loudness is left out (peak_db/ref_db None) so only the text
decides, except where a test checks the loudness gates.
"""
import pytest

from turn_rules import decide, normalize


def stops(text, agent=""):
    return decide(text, agent)[0]


# --- floor bids that must STOP ------------------------------------------------------------------------------------------
STOP_PHRASES = [
    # "wait" in its dialect forms
    ("انطر شوي", "Hijazi / Eastern / Kuwaiti: انطر = wait"),
    ("نطر نطر", "Kuwaiti / Bahraini: نطر = wait"),
    ("انطري لحظة", "Hijazi, to a female agent"),
    ("لحضة لو سمحت", "Gulf spelling: ض for ظ (the two sounds merged in Gulf speech)"),
    ("تواني بس", "urban Hijazi: ث pronounced ت (ثواني -> تواني)"),
    ("تقيقة بس", "ASR / speaker voicing: د heard as ت (دقيقة -> تقيقة)"),
    # "excuse me" / permission to talk
    ("لو سمحت ممكن", "all regions: excuse me"),
    ("لو سمحتي اختي", "all regions, to a female agent"),
    ("اسمح لي اقاطعك", "Najdi / general: allow me"),
    ("عن اذنك", "general: excuse me"),
    ("معليش", "Gulf / Hijazi: sorry, excuse me (the dialect form of «عفوا»)"),
    ("معلش يا اخي", "Hijazi / Egyptian-influenced: excuse me"),
    ("كفاية كذا", "Hijazi: that is enough"),
    ("شوي شوي علي", "Gulf: slow down"),
    ("على مهلك", "general: slow down"),
    # negation / correction opening the turn
    ("مو هذا قصدي", "Najdi / Gulf negation at the start"),
    ("مب كذا الكلام", "Emirati / Qatari: مب"),
    ("موب صحيح هالكلام", "Najdi: موب"),
    ("مهوب هذا الموعد", "Najdi: مهوب"),
    ("مش هيك", "Hijazi / Levantine-influenced: مش"),
    ("ماهو صحيح", "Najdi / Hijazi: ماهو"),
    ("قصدي الاسبوع الجاي", "correction at the start: I mean"),
    ("إيه بس مو هذا", "ack then a correction"),
    # dialect question words (two or more words, loudness unknown -> STOP)
    ("امتى يوصل", "Hijazi: امتى = when"),
    ("فين الفرع", "Hijazi: فين = where"),
    ("منو انت", "Kuwaiti / Bahraini: منو = who"),
    ("شكثر المبلغ", "Kuwaiti: شكثر = how much"),
    ("بكم الاشتراك", "general: how much"),
    ("اشلون يعني", "Kuwaiti / Iraqi-influenced: how"),
    # phrases
    ("ابغى اعرف ليش", "Gulf: I want to know"),
    ("عندي استفسار", "general: I have a question"),
]

# --- must NOT stop (backchannels, agreement, polite "no problem") --------------------------------------------------------
KEEP_PHRASES = [
    ("إيه", "ack"), ("أيوه", "ack"), ("تمام", "ack"), ("زين", "ack"), ("طيب", "ack"),
    ("إيه إيه", "ack"), ("أيوه أيوه تمام", "ack"), ("زين زين", "ack"), ("ماشي", "ack"), ("هيه", "Kuwaiti ack"),
    ("إي والله", "ack"), ("صح صح", "ack"), ("اها", "ack"), ("مممم", "filler"),
    ("مو مشكلة", "no problem = agreement"),
    ("مب مشكلة", "Emirati: no problem"),
    ("مش مشكلة", "Hijazi: no problem"),
    ("مو بطال", "not bad = agreement"),
    ("مب شين", "Emirati: not bad"),
    ("مو غلط", "not wrong = agreement"),
    ("مو", "a lone negation particle is too little to tell"),
    ("ايه معك", "ack"),
    ("الله يعطيك العافية", "thanks"),
    ("ترجمة نانسي قنقر", "Whisper stock phrase on noise"),
]


@pytest.mark.parametrize("text,why", STOP_PHRASES)
def test_dialect_floor_bids_stop(text, why):
    assert stops(text), f"should STOP: «{text}» ({why}); got {decide(text)}"


@pytest.mark.parametrize("text,why", KEEP_PHRASES)
def test_backchannels_keep(text, why):
    assert not stops(text), f"should KEEP: «{text}» ({why}); got {decide(text)}"


def test_negation_mid_sentence_is_not_a_cue():
    # «مو» in the middle of a sentence is ordinary speech; only the 3-word rule may stop it, never the start-cue rule
    assert decide("الموعد مو اليوم")[1] != "cue «مو»"


def test_normalisation_merges_gulf_letters():
    assert normalize("لحضة") == normalize("لحظة")             # ض / ظ merged in Gulf speech
    assert normalize("ثواني") == normalize("ثواني")


def test_new_cues_respect_far_quieter_gate():
    # TV / echo far under the caller's voice: even a cue does not count (same gate as the old cues)
    assert not decide("لو سمحت", peak_db=-35, ref_db=-15)[0]
    assert not decide("معليش", peak_db=-35, ref_db=-15)[0]


def test_question_words_respect_quiet_gate():
    assert not decide("امتى يوصل", peak_db=-25, ref_db=-15)[0]   # 10 dB under the caller: a question, but quiet
    assert decide("امتى يوصل", peak_db=-15, ref_db=-15)[0]


def test_not_cues_still_blocked():
    assert not stops("أستاذ")                                    # one letter from «استنا»
    assert not stops("وأني")                                     # one letter from «ثواني»
