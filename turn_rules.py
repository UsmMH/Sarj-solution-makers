"""Text rules: given what the caller seems to have said (from the ASR), should the agent stop?

Pure Python: no audio and no model, so it can be tested offline (see eval_rules.py).

    decide("لا لا لحظة")        -> (True,  'cue «لحظه»')
    decide("إيه إيه")            -> (False, 'acknowledgement or too little to tell')
    decide("ترجمة نانسي قنقر")   -> (False, 'stock phrase the ASR invents on noise')

Evidence, in this order:
  1. things that mean "not a bid": nothing heard, Whisper's stock phrases, an echo of the agent's own words,
     talk to someone else in the room;
  2. strong stop cues (wait / no / listen / but) and questions;
  3. otherwise: two or more words that are not acknowledgements or fillers = the caller is saying something.

The word lists come from general Gulf and Modern Standard Arabic plus the train rows of scenarios_2.csv.
Some entries were added after reading the dev transcripts, so a dev score for this file is a sanity check,
not an estimate of how it does on audio it has never seen.
"""
import re
from difflib import SequenceMatcher

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")   # tashkeel, tatweel
_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def normalize(text: str) -> str:
    """Strip vowel marks and punctuation, unify letter variants, collapse elongation (اهوووو -> اهو)."""
    t = _DIACRITICS.sub("", text or "").translate(_LETTERS).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return " ".join(t.split())


def _words(s: str) -> frozenset:
    return frozenset(normalize(w) for w in s.split())


# --- word lists (written in natural spelling, normalised on load) -----------------------------------------------
STOP_CUES = _words("""
    لحظة لحظات ثواني ثانية دقيقة استنى استنا استنوا انتظر انتظري وقف توقف قف
    اسمع اسمعني اسمعي اسمعيني بس عفوا سوري sorry wait stop pardon
""")
PREFIX_CUES = ("لحظ", "استن")            # a word still being said: «لحظ…», «استن…»
QUESTION_WORDS = _words("كم شنو وش ايش ليش ليه وين متى شلون كيف مين لماذا ماذا هل شو اش")
SIDE_TALK = _words("يمه يما ماما بابا يابه اسكت اسكتي")          # talking to someone in the room
SIDE_TALK_PHRASE = re.compile(r"(?:^| )يا (?:ولد|ولدي|بنت)(?: |$)")
ACKS = _words("""
    ايه ايوه ايوا ايو اي ايها اها اه اوه ام امم مم م هم همم هيه هي ها هاه
    اوكي اوك اوكيه تمام زين ماشي طيب خلاص حلو كويس ممتاز مميز مميزة رائع جيد
    صح صحيح صادق بالضبط اكيد طبعا فعلا نعم اجل يب
    والله الله بالله الحمدلله شكرا يعطيك العافية اهلا هلا مرحبا
    معايا معاك سمعتك فاهم فاهمة فهمت عليك
""")
NEUTRAL = _words("يا و ثم")
STOCK = re.compile(r"(?:^| )(?:ترجمه|اشترك\w*|المشاهده|مشاهده)(?: |$)")      # what Whisper invents on noise
LAUGH = re.compile(r"(?:ه|ها)+")
WEAK_EVIDENCE_MIN_DB = -20.0   # fallback only (no reference yet): absolute peak; fails when the whole line is quieter
WEAK_EVIDENCE_REL_DB = -6.0    # peak vs the caller's own earlier voice: real stops -2..+6 dB, TV / echo / background -11..-18 dB
MIN_CONTENT_WORDS = 3     # two odd words are what Whisper makes of noise («في بوح», «يوم ما»); three is real speech


def tokens(text: str) -> list:
    out = []
    for t in normalize(text).split():
        out += ["لا"] * (len(t) // 2) if re.fullmatch(r"(?:لا)+", t) else [t]     # «لالا» -> لا لا
    return out


def _close(a: str, b: str) -> bool:
    return a == b or (len(a) >= 3 and len(b) >= 3 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def is_ack(tok: str) -> bool:
    return tok in ACKS or bool(LAUGH.fullmatch(tok)) or any(_close(tok, a) for a in ACKS if len(a) >= 3)


def find_cue(toks: list):
    for i, t in enumerate(toks):
        if t == "لا" and (i == 0 or (i == 1 and is_ack(toks[0]))):   # «لا» at the start, or «إيه… لا»; not mid-sentence
            return t
        if t in STOP_CUES or (len(t) >= 3 and t.startswith(PREFIX_CUES)):
            return t
        if len(t) >= 4 and any(len(c) >= 4 and SequenceMatcher(None, t, c).ratio() >= 0.8 for c in STOP_CUES):
            return t
    return None


def looks_like_echo(toks: list, agent_text: str) -> bool:
    """Most of what was 'heard' is the agent's own sentence leaking back into the caller channel."""
    if not agent_text or len(toks) < 3:
        return False
    agent = normalize(agent_text).split()
    hits = sum(1 for t in toks if any(_close(t, a) for a in agent))
    return hits / len(toks) >= 0.7


# --- end of turn: has the caller finished, or only paused? -------------------------------------------------------------
# Whisper keeps a trailing «...» or «،» when a sentence hangs, and a last word like «و», «يعني», «في» or «اممم» means
# something is still coming. This only decides how long to wait; it never makes the agent take the turn earlier.
EOT_FILLERS = _words("ا ام امم مم م هم همم اه اوه")
EOT_OPEN_WORDS = _words("""
    و ف ب ل ثم او لكن بس لان علشان عشان اللي الي الذي التي ان انه انها
    في من على الى عن مع حتى زي مثل مثلا يعني اقصد شسمه شوف
""")
EOT_OPEN_END = ("...", "…", "،", ",", "-", "—")


def eot_incomplete(text: str):
    """(incomplete?, reason). True = the caller probably paused mid-thought: wait longer before taking the turn."""
    raw = (text or "").strip()
    toks = tokens(raw)
    if not toks:
        return False, "nothing heard"
    if STOCK.search(" ".join(toks)):
        return False, "stock phrase the ASR invents on noise"
    if raw.endswith(EOT_OPEN_END):
        return True, "trailing «...» or comma"
    last = toks[-1]
    if last in EOT_FILLERS:
        return True, f"ends on filler «{last}»"
    if last in EOT_OPEN_WORDS:
        return True, f"ends on «{last}»"
    return False, "sounds finished"


def decide(text: str, agent_text: str = "", peak_db=None, ref_db=None):
    """(stop?, reason). False means 'no reason to stop yet': a longer burst may still be decoded again.

    peak_db = loudest 20 ms of the burst (dBFS), if known. Only the weak paths (a question, or several unremarkable
    words) need it: a quiet burst that Whisper turned into «ماذا شعب؟» is far more likely TV, echo or someone else
    in the room than the caller. Stop cues are never gated.
    ref_db = loudest 20 ms of the caller's own voice earlier in this call. With it, "quiet" means quieter than the
    caller usually is, so a softer speaker or a quieter phone line no longer looks like TV. Without it, the old
    absolute -20 dB line is used.
    """
    toks = tokens(text)
    norm = " ".join(toks)
    if not toks:
        return False, "nothing heard"
    if STOCK.search(norm):
        return False, "stock phrase the ASR invents on noise"
    if looks_like_echo(toks, agent_text):
        return False, "echo of the agent's own words"
    if any(t in SIDE_TALK for t in toks) or SIDE_TALK_PHRASE.search(norm):
        return False, "talking to someone else in the room"
    cue = find_cue(toks)
    if cue:
        return True, f"cue «{cue}»"
    if peak_db is None:
        quiet = False
    elif ref_db is not None:
        quiet = peak_db - ref_db < WEAK_EVIDENCE_REL_DB
    else:
        quiet = peak_db < WEAK_EVIDENCE_MIN_DB
    if len(toks) >= 2 and any(t in QUESTION_WORDS for t in toks):
        return (False, "question, but quiet") if quiet else (True, "question")
    content = list(dict.fromkeys(t for t in toks if t not in NEUTRAL and not is_ack(t)))   # distinct words
    if len(content) >= MIN_CONTENT_WORDS:
        if quiet:
            return False, f"{len(content)} words, but quiet"
        return True, f"{len(content)} words that are not acknowledgements"
    return False, "acknowledgement or too little to tell"


def worth_second_opinion(text: str, why: str, allow_quiet: bool = False) -> bool:
    """decide() said keep, but not for a sure reason: ask the LLM (llm_judge.py).

    Sure keeps (nothing heard, stock phrase, echo, side talk) and plain acknowledgements are never asked.
    Quiet bursts are asked only with allow_quiet: on the dev data they were all TV, echo or noise.
    """
    if why == "question, but quiet" or why.endswith("words, but quiet"):
        return allow_quiet
    if why != "acknowledgement or too little to tell":
        return False
    return any(t not in NEUTRAL and not is_ack(t) for t in tokens(text))