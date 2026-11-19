"""Live demo: talk to the agent through your microphone and watch our system decide, in your browser.

    python demo.py                 # loads Whisper, then open http://127.0.0.1:8780 in Chrome or Edge
    python demo.py --port 8780

The page plays the real Arabic agent sentences of a dev call. Your microphone goes to the SAME MySystem class the
benchmark talks to, 20 ms at a time. STOP cuts the agent's voice at once; TAKE_TURN (you finished) makes the agent
go on with its next sentence. Every Whisper decode, its decision and the loudness gate are shown live.

Use headphones: from speakers, the agent's own voice reaches the microphone (the system then hears an echo).
Nothing leaves this laptop: the page, the audio and Whisper all stay on 127.0.0.1.
"""
import argparse
import asyncio
import base64
import collections
import http
import http.client
import io
import json
import os
import queue as sentence_queue                          # "queue" is the page-message queue inside handler()
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from difflib import SequenceMatcher

import my_system
from llm_judge import _read_key
from turn_rules import _close, decide, eot_incomplete, normalize, tokens
from my_system import ASR, FLOOR_PERCENTILE, LOUD_DB, MySystem, loudness_db
from ttbench.dataset import read_wav_stereo

# Demo only (this process; the scored my_system.py is untouched): a person talking freely pauses to think, so the demo
# waits longer before taking the turn than the benchmark baseline (700 ms). The reply draft still starts at 300 ms of
# silence and is ready after ~2 s anyway, so a later TAKE_TURN costs almost no reply time.
my_system.TAKE_TURN_AFTER_MS = int(os.environ.get("DEMO_TURN_WAIT_MS", "1500"))    # text sounds finished (was 1200 on Nov 19)
my_system.EOT_HOLD_MS = int(os.environ.get("DEMO_EOT_HOLD_MS", "3200"))            # text ends mid-thought: «يوم الـ..», «آآ،»

HERE = Path(__file__).parent
DATA = HERE / "data" / "dev"
PAGE = HERE / "demo" / "index.html"
REC_DIR = HERE / "demo_recordings"                  # what the server really received from the mic (git-ignored)
SR = 16000


def load_calls():
    """Every dev call with its agent sentences (text + where they sit in the agent channel)."""
    calls = []
    for path in sorted((DATA / "labels").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        utts = [{"id": u["utt_id"], "text": u["text"], "start_ms": u["start_ms"], "end_ms": u["end_ms"]}
                for u in d.get("agent_utterances", [])]
        if utts:
            calls.append({"id": d["scenario_id"], "use_case": d.get("context", {}).get("use_case", ""),
                          "category": d.get("category", ""), "file": d["audio"]["file"], "utterances": utts})
    return calls


CALLS = {c["id"]: c for c in load_calls()}
_clip_cache = {}


def agent_clip(call_id, utt_id):
    """The agent's own recording of one sentence, as a mono WAV file (bytes)."""
    key = (call_id, utt_id)
    if key not in _clip_cache:
        c = CALLS[call_id]
        u = next(u for u in c["utterances"] if u["id"] == utt_id)
        audio = read_wav_stereo(DATA / "audio" / c["file"])[u["start_ms"] * 16: u["end_ms"] * 16, 0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(audio.astype("<i2").tobytes())
        _clip_cache[key] = buf.getvalue()
    return _clip_cache[key]


def respond(status, body, ctype):
    headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
    return Response(status.value, status.phrase, headers, body)


def process_request(connection, request):
    """Plain HTTP for the page and the agent clips; /ws is upgraded to the live websocket."""
    path = request.path.split("?")[0]
    if path == "/ws":
        return None
    if path in ("/", "/index.html"):
        return respond(http.HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
    if path == "/calls":
        body = json.dumps([{k: c[k] for k in ("id", "use_case", "category", "utterances")} for c in CALLS.values()],
                          ensure_ascii=False).encode("utf-8")
        return respond(http.HTTPStatus.OK, body, "application/json; charset=utf-8")
    if path.startswith("/clip/"):
        try:
            _, _, call_id, utt = path.split("/")
            return respond(http.HTTPStatus.OK, agent_clip(call_id, utt.removesuffix(".wav")), "audio/wav")
        except (ValueError, KeyError, StopIteration):
            pass
    return respond(http.HTTPStatus.NOT_FOUND, b"not found", "text/plain")


CHAT_MODEL = os.environ.get("DEMO_LLM", "anthropic/claude-haiku-5.5")   # $0.10 / $0.50 per M tokens: ~$0.0002 a reply
CHAT_MAX_HISTORY = 12                               # only the last 12 lines go to the LLM: a long call costs no more per reply
# Written in Arabic on purpose (Nov 18): with English instructions the lines drifted to Modern Standard Arabic
# («هل يناسبك؟», «أود فقط»); this version made Haiku sound like a real Saudi agent («أبشر، نقدر نرتبه لك»).
CHAT_PROMPT = """أنت موظفة خدمة عملاء سعودية في مكالمة جوال حقيقية. أنتِ اللي اتصلتي، وهدف المكالمة (التزمي فيه طول المكالمة): {scenario}

طريقة كلامك:
- لهجة سعودية بيضاء طبيعية مثل ما يتكلم الموظف فعلاً بالجوال، مو فصحى: قولي «يناسبك؟» مو «هل يناسبك؟»،
  «أبي أتأكد» مو «أود التأكد»، «بكرة» مو «غداً». لا تستخدمي: هل، سوف، لقد، حضرتك، عزيزي العميل.
- ادخلي في الموضوع على طول. عبارة مجاملة وحدة بالكثير في الرد (أبشر، تمام، ولا يهمك…)، وأغلب الردود بدونها.
  لا تكررين «الله يعافيك» و«أبشر» و«شكراً» في كل رد، ولا تعيدين كلام العميل ولا تأكدين نفس الشي مرتين.
- ردود قصيرة جداً: جملة وحدة غالباً، أقل من 12 كلمة. حروف عربية فقط، بدون حروف لاتينية أو رموز أو أقواس.
- الختام قصير: أكدي الموعد بجملة وحدة ثم «مع السلامة».
- كلام العميل جاي من نظام تعرّف صوت وممكن يكون فيه أخطاء؛ إذا ما فهمتي، اطلبي منه يعيد بلطف.
- إذا جملتك السابقة تنتهي بـ «…» يعني العميل قاطعك هنا: ردي على كلامه، لا تعيدين جملتك.
- إذا كلام العميل «…» يعني سكت وما رد: اسأليه بلطف إذا يسمعك.
- ما تعرفين إذا العميل رجّال أو حرمة، فلا تخاطبينه بصيغة مذكر ولا مؤنث («تفضّل/تفضّلي»، «معك/معكِ»)؛
  استخدمي صيغ تناسب الاثنين («وش السؤال؟»، «يناسبك؟»)، إلا إذا عرفتي جنسه من كلامه.
- اسمك سارة. قولي دورك فقط ثم وقفي، لا تجاوبين عن العميل ولا تخترعين أسماء ثانية.
- إذا خلصت المهمة ودّعيه وختمي بـ «مع السلامة»."""
DEFAULT_SCENARIO = ("عيادة أسنان تذكّر العميل بموعده يوم الأحد الساعة عشرة الصبح وتسأله إذا يقدر يحضر؛ "
                    "إذا ما يناسبه اعرض عليه الاثنين أو الثلاثاء العصر.")


_STOCK_PHRASES = re.compile(r"ترجمة نانسي قنقر|اشتركوا في القناة|شكرا للمشاهدة")


def clean_turn(text):
    """Whisper's repeat loops («حالوووووو…», a word 10 times) and stock noise phrases out of a turn's text."""
    t = _STOCK_PHRASES.sub(" ", text or "")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                     # a letter at most twice in a row
    t = re.sub(r"(\S+)(?:[\s،,]+\1){2,}", r"\1 \1", t)        # a word at most twice in a row
    return " ".join(t.split())


def _from_user(msgs):
    """A cut history must still start with the caller's turn, as chat models expect."""
    return msgs[1:] if msgs and msgs[0]["role"] == "assistant" else msgs


def chat_reply(history, scenario):
    """One agent line from the LLM (OpenRouter), or None. Blocking: run it in a thread."""
    key = _read_key()
    if not key:
        return None, "no OPENROUTER_API_KEY (.env)"
    body = json.dumps({"model": CHAT_MODEL, "temperature": 0.6, "max_tokens": 160, "reasoning": {"enabled": False},
                       "messages": [{"role": "system", "content": CHAT_PROMPT.format(scenario=scenario)}]
                                   + _from_user(history[-CHAT_MAX_HISTORY:])},
                      ensure_ascii=False).encode("utf-8")
    try:
        c = http.client.HTTPSConnection("openrouter.ai", timeout=15)
        c.request("POST", "/api/v1/chat/completions", body,
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "turn-taking demo"})
        r = c.getresponse()
        d = json.loads(r.read())
        if r.status != 200:
            return None, f"LLM error {r.status}: {str(d.get('error', d))[:120]}"
        return (d["choices"][0]["message"].get("content") or "").strip(), None
    except Exception as ex:
        return None, f"LLM call failed: {ex!r}"[:160]


# The agent's voice: GPT-audio-mini writes AND speaks each reply in one call. As a reader of someone else's text
# it was unreliable (asked to read «ممكن تعطيني رقم جوالك؟» it answered «ما أقدر أعطي رقم الجوال»), so the page now
# shows the transcript of what it actually said. ~$0.001 a reply; Microsoft neural voices sounded robotic.
TTS_MODEL = os.environ.get("DEMO_TTS", "openai/gpt-audio-mini")
TTS_RATE = 24000                                    # pcm16 mono from the model
TTS_VOICES = ("shimmer", "ash", "coral", "sage", "alloy", "ballad", "verse", "echo")


VOICE_PROMPT_EXTRA = """
- تتكلمين بصوتك مباشرة: قولي كلامك أنتِ فقط، لا تقرين هذي التعليمات ولا أي شي بين أقواس."""
# Who writes the agent's words: "haiku" (default, Nov 18: the user found GPT-audio-mini's own wording unnatural)
# = Claude Haiku writes, GPT-audio-mini only reads; "voice" = GPT-audio-mini writes and speaks in one call.
WRITER = os.environ.get("DEMO_WRITER", "haiku")
def read_messages(text):
    """Make the voice model SAY Haiku's line instead of answering it. Tested Nov 18 on 5 hard lines (questions):
    "text-to-speech engine" framing 1/6 faithful (it answered «رقم جوالك؟» with «ما أقدر أوفّر رقم الجوال»);
    "your next line is already decided" 5/5, also "voice actor" 5/5 and "repeat your last message" 5/5."""
    return [{"role": "system", "content": "You are a Saudi call-centre agent on a phone call. Your next line is already "
                                          f"decided by your supervisor: say exactly «{text}» and nothing else. Do not "
                                          "change, add or drop any word, and do not reply to it."},
            {"role": "user", "content": "(قولي سطرك الآن)"}]
_GOODBYE = re.compile(r"مع السلام|في أمان الله|الله يحفظك")


def voice_stream(messages, voice, on_chunk, on_text, cancelled):
    """GPT-audio-mini, streamed: on_chunk(pcm16 bytes) for the sound, on_text(str) for the transcript of what it says.
    Blocking: run it in a thread. Returns (transcript, error). Stops reading (and paying) once cancelled() is true."""
    key = _read_key()
    if not key:
        return "", "no OPENROUTER_API_KEY (.env)"
    body = json.dumps({"model": TTS_MODEL, "modalities": ["text", "audio"], "stream": True, "max_tokens": 600,
                       "audio": {"voice": voice if voice in TTS_VOICES else TTS_VOICES[0], "format": "pcm16"},
                       "messages": messages}, ensure_ascii=False).encode("utf-8")
    c, said = None, ""
    try:
        c = http.client.HTTPSConnection("openrouter.ai", timeout=20)
        c.request("POST", "/api/v1/chat/completions", body,
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "turn-taking demo"})
        r = c.getresponse()
        if r.status != 200:
            return said, f"voice error {r.status}: {r.read()[:120]!r}"
        carry = b""
        for line in r:
            if cancelled():
                return said, None
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if "error" in d:
                return said, f"voice error: {str(d['error'])[:120]}"
            for ch in d.get("choices", []):
                audio = (ch.get("delta") or {}).get("audio") or {}
                if audio.get("transcript"):
                    said += audio["transcript"]
                    on_text(said)
                if audio.get("data"):
                    pcm = carry + base64.b64decode(audio["data"])
                    cut = len(pcm) - len(pcm) % 2                  # whole 16-bit samples only
                    carry = pcm[cut:]
                    if cut:
                        on_chunk(pcm[:cut])
        return said, None
    except Exception as ex:
        return said, f"voice failed: {ex!r}"[:160]
    finally:
        if c is not None:
            c.close()


# Optional cloud transcription of the caller's FINISHED turn (demo only; the fast STOP decisions stay on local Whisper).
# Tested Nov 18 on a noisy-room call: Gemini 3.7 Flash cut its answers off (its hidden reasoning used the token budget);
# Gemini 2.5 Flash without reasoning: complete text in ~2 s for ~$0.00016 a turn. Qwen 3.8 omni added unsaid sentences.
STT_MODEL = os.environ.get("DEMO_STT_MODEL", "google/gemini-2.5-flash")
STT_TIMEOUT_S = 6.0
STT_PROMPT = ("Transcribe this Arabic phone audio (Saudi / Gulf dialect) exactly as spoken, in Arabic script. It may "
              "start or end mid-word. Output only the transcript. If there is no speech, output nothing.")


def cloud_transcribe(audio):
    """-> (text, ms, error). Blocking: run it in a thread. audio = float32 at 16 kHz."""
    key = _read_key()
    if not key:
        return "", 0, "no OPENROUTER_API_KEY (.env)"
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    body = json.dumps({"model": STT_MODEL, "temperature": 0, "max_tokens": 300, "reasoning": {"enabled": False},
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": STT_PROMPT},
                           {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(),
                                                                   "format": "wav"}}]}]}).encode()
    t0 = time.perf_counter()
    try:
        c = http.client.HTTPSConnection("openrouter.ai", timeout=STT_TIMEOUT_S)
        c.request("POST", "/api/v1/chat/completions", body,
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "turn-taking demo"})
        r = c.getresponse()
        d = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        if r.status != 200:
            return "", ms, f"cloud STT error {r.status}: {str(d.get('error', d))[:100]}"
        return (d["choices"][0]["message"].get("content") or "").strip(), ms, None
    except Exception as ex:
        return "", (time.perf_counter() - t0) * 1000, f"cloud STT failed: {ex!r}"[:120]


# --- ElevenLabs (Nov 18 evening): the fast voice and the best transcription in noise -------------------------------
# Voice: built-in voices with eleven_v3_conversational, first sound ~0.5 s (GPT-audio-mini ~1.3 s). Free accounts cannot use
# Voice Library voices (Saudi ones) through the API (402 paid_plan_required). Scribe v1 on a noisy laptop-mic call:
# «دقيقة ما أسمعك عيدها مرة ثانية…» where Whisper heard «دقيقة ما اسمع كايدين مالساني», ~1.1-2 s per turn.
ELEVEN_VOICES = {"eleven-sarah": "EXAVITQu4vr4xnSDxMaL", "eleven-chris": "iP95p4xoKVk53GoZ742B",
                 "eleven-sarah-quick": "EXAVITQu4vr4xnSDxMaL", "eleven-chris-quick": "iP95p4xoKVk53GoZ742B",
                 # Saudi Voice Library voices (paid plan, Nov 19): they speak through the API without being added to the account
                 "saudi-abir": "VcSFj5ljnKA0aQcxeHpl", "saudi-hala": "2aVe3IK7dgUBaUTdfarw",
                 "saudi-maryam": "aMmeBf0lzDYlouyfqNjh", "saudi-jumana": "4uJbyDJGijiDphW2T3vP"}
# "quick" voices (Nov 19): v4 Turbo and v3 ignore the speed setting (same 4.39 s sentence at 1.0 and 1.2); Flash v2.5
# honours it and is quicker anyway (3.17 s at 1.0, 2.36 s at 1.2). Same 0.5 credit per character.
ELEVEN_STYLE = {"eleven-sarah-quick": ("eleven_flash_v2_5", 1.01), "eleven-chris-quick": ("eleven_flash_v2_5", 1.01)}
QUICK_SPEED = float(os.environ.get("DEMO_QUICK_SPEED", "1.1"))      # the Saudi "quick" voices (Flash v2.5 honours speed)
for _k in [k for k in ELEVEN_VOICES if k.startswith("saudi-")]:
    ELEVEN_VOICES[_k + "-quick"] = ELEVEN_VOICES[_k]
    ELEVEN_STYLE[_k + "-quick"] = ("eleven_flash_v2_5", QUICK_SPEED)
# Munsit (Faseeh) native Saudi voices, "munsit:" + voice id (Nov 19): speak through munsit_speak() below
ELEVEN_VOICES.update({"munsit-nora": "munsit:sa-nora-vo-f", "munsit-maha": "munsit:IPK8qQ3F5NMiQLWFz1a83TG3",
                      "munsit-hala": "munsit:08XOzRjaaumxbHhcGOrWkJ7z", "munsit-lama": "munsit:ar-hijazi-female-1",
                      "munsit-noura": "munsit:ar-najdi-female-1"})
# no "quick" Munsit voices: voice_settings.speed 1.1 left the audio length unchanged over REST (3.4 s both, Nov 19)
# Nov 19: eleven_v4_turbo (free tier OK, same 0.5 credit/char, first sound ~0.5 s like v3) chosen by ear; it also
# accepts previous_text, so the sentence pieces keep one intonation. DEMO_ELEVEN_MODEL=eleven_v3_conversational = old voice.
ELEVEN_MODEL = os.environ.get("DEMO_ELEVEN_MODEL", "eleven_v4_turbo")
ELEVEN_RATE = 16000
DRAFT_AFTER_MS = 300     # caller quiet this long: start drafting the reply (TAKE_TURN itself comes at 700 ms or later)
_SENTENCE_END = re.compile(r"[.!?؟،,]\s")


def _env(name):
    """A key from this window's environment or the git-ignored .env file."""
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        for line in (HERE / ".env").read_text(encoding="utf-8-sig").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip("'\"")
    except OSError:
        pass
    return ""


def scribe_transcribe(audio):
    """ElevenLabs Scribe v1 on a finished turn -> (text, ms, error). Blocking: run it in a thread."""
    key = _env("ELEVENLABS_API_KEY")
    if not key:
        return "", 0, "no ELEVENLABS_API_KEY (.env)"
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    bd = os.urandom(12).hex()
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\nscribe_v1\r\n"
            f"--{bd}\r\nContent-Disposition: form-data; name=\"language_code\"\r\n\r\nara\r\n"
            f"--{bd}\r\nContent-Disposition: form-data; name=\"tag_audio_events\"\r\n\r\nfalse\r\n"
            f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"turn.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{bd}--\r\n".encode()
    t0 = time.perf_counter()
    try:
        c = http.client.HTTPSConnection("api.elevenlabs.io", timeout=STT_TIMEOUT_S)
        c.request("POST", "/v1/speech-to-text", body, {"xi-api-key": key, "Content-Type": f"multipart/form-data; boundary={bd}"})
        r = c.getresponse()
        d = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        if r.status != 200:
            return "", ms, f"Scribe error {r.status}: {str(d.get('detail', d))[:100]}"
        text = re.sub(r"[　-鿿]", "", d.get("text", ""))     # it once slipped a Chinese character in
        return text.strip(), ms, None
    except Exception as ex:
        return "", (time.perf_counter() - t0) * 1000, f"Scribe failed: {ex!r}"[:120]


# Writers for the free call (Nov 18 test, same 4 turns): Haiku first sentence 1.33 s, warm natural Saudi wording;
# Gemini 2.5 Flash-Lite 1.02 s but sloppier (invented a clinic name, «معكِ» to a man, skipped a question).
WRITERS = {"haiku": "anthropic/claude-haiku-5.5", "flash-lite": "google/gemini-2.5-flash-lite"}


class ScribeLive:
    """ElevenLabs real-time Scribe (scribe_v3_live) for the caller's words, streamed while they speak.

    Tested Nov 18 on a recorded noisy-room turn: scribe_v3_live wrote «دقيقة، دقيقة الاثنين ولا الثلاثاء؟ ما سمعتيني
    زين» word for word (Whisper: «جيكة جيكة… سينا»), final text 253 ms after a manual commit (batch Scribe ~1.2 s).
    The mic is sent every 100 ms; at the caller's pause we commit and get their exact words back almost at once.
    Turn text = everything committed since the agent last took the floor."""
    URL = ("wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v3_live&audio_format=pcm_16000"
           "&language_code=ar&commit_strategy=manual")

    def __init__(self, on_partial):
        self.ws = None
        self.ok = False
        self.buf = []
        self.turn = []                                  # committed pieces of the caller's current turn
        self.partial = ""
        self.on_partial = on_partial
        self._commit_waiter = None
        self.agent_lines, self.agent_words = [], set()
        self.recent = collections.deque(maxlen=50 * 6)  # the last 6 s of mic frames, to send a held-back burst
        self.held = False
        self._zeros = np.zeros(320, dtype="<i2").tobytes()

    async def start(self):
        from websockets.asyncio.client import connect as ws_connect
        key = _env("ELEVENLABS_API_KEY")
        if not key:
            return "no ELEVENLABS_API_KEY (.env)"
        try:
            self.ws = await asyncio.wait_for(ws_connect(self.URL, additional_headers={"xi-api-key": key},
                                                        max_size=2 ** 22), 8)
            first = json.loads(await asyncio.wait_for(self.ws.recv(), 8))
            if first.get("message_type") != "session_started":
                return f"live Scribe refused: {str(first)[:120]}"
            self.ok = True
            asyncio.create_task(self._reader())
            return None
        except Exception as ex:
            return f"live Scribe failed to connect: {ex!r}"[:150]

    async def _reader(self):
        try:
            async for raw in self.ws:
                m = json.loads(raw)
                kind = m.get("message_type", "")
                if kind == "partial_transcript":
                    self.partial = m.get("text", "")
                    self.on_partial(" ".join(self.turn + [self.partial]).strip())
                elif kind.startswith("committed_transcript"):
                    if m.get("text", "").strip():
                        self.turn.append(m["text"].strip())
                    self.partial = ""
                    if self._commit_waiter and not self._commit_waiter.done():
                        self._commit_waiter.set_result(True)
                elif kind not in ("session_started",):
                    if self._commit_waiter and not self._commit_waiter.done():
                        self._commit_waiter.set_result(False)     # an error: do not keep a commit waiting
        except Exception:
            pass
        self.ok = False

    def feed(self, frame, hold=False, burst_frames=0):
        """One 20 ms caller frame (int16). Sent in 100 ms messages.

        hold = the agent is talking and our engine has not stopped it. On speakers the mic then hears the agent, and
        Scribe wrote its lines into the caller's turn, even short ones the text filter cannot tell apart («هلا والله»,
        «يناسبك؟», Nov 19). So while held, Scribe gets silence. When the hold ends (the engine said STOP, or the agent
        finished while the caller was mid-word), the caller's burst is sent from 400 ms before its start (burst_frames = its
        length in frames, even when it already ended): the words that made us stop are never lost."""
        if not self.ok:
            return
        self.recent.append(frame)
        if hold:
            self.held = True
            self.buf.append(self._zeros)
        elif self.held:
            self.held = False
            n = min(burst_frames + 20, len(self.recent)) if burst_frames else 1  # + 400 ms before it, as Whisper gets
            self.buf += [f.tobytes() for f in list(self.recent)[-n:]]
        else:
            self.buf.append(frame.tobytes())
        if len(self.buf) >= 5:
            data, self.buf = b"".join(self.buf), []
            asyncio.create_task(self._send(data, False))

    async def _send(self, data, commit):
        try:
            await self.ws.send(json.dumps({"message_type": "input_audio_chunk", "commit": commit, "sample_rate": SR,
                                           "audio_base_64": base64.b64encode(data).decode()}))
        except Exception:
            self.ok = False

    async def commit(self, timeout=1.5):
        """The caller paused: finalise what they said. -> the turn's text so far ("" if the stream failed)."""
        if not self.ok:
            return ""
        data, self.buf = b"".join(self.buf), []
        self._commit_waiter = asyncio.get_running_loop().create_future()
        await self._send(data, True)
        try:
            await asyncio.wait_for(self._commit_waiter, timeout)
        except asyncio.TimeoutError:
            pass
        return self.turn_text()

    def turn_text(self):
        """The caller's words of this turn, minus the agent's own voice. On speakers the mic hears the agent and live
        Scribe wrote its lines into the caller's turn («هلا والله، معك سارة…», Nov 18): drop every clause that is mostly
        the agent's recent words."""
        text = " ".join(self.turn + ([self.partial] if self.partial else [])).strip()
        if not self.agent_words:
            return text
        kept = []
        for clause in re.split(r"(?<=[.،,؟?!])\s+", text):
            toks = tokens(clause)
            # an echo is a long verbatim stretch of the agent's last line; a caller repeating a few of its words
            # («الثلاثاء العصر؟», «ما في الأربعاء؟») is not, so short clauses always stay
            if len(toks) >= 4 and sum(any(_close(t, a) for a in self.agent_words) for t in toks) / len(toks) >= 0.7:
                continue
            kept.append(clause)
        return " ".join(kept).strip()

    def agent_said(self, text):
        """Remember the agent's last lines (their words), so their echo can be removed from the caller's turn."""
        self.agent_lines = (self.agent_lines + [text])[-1:]           # the line that can echo is the last one
        self.agent_words = {t for line in self.agent_lines for t in tokens(line)}

    def new_turn(self):
        """The agent took the floor: the next words belong to the caller's next turn."""
        self.turn, self.partial = [], ""

    async def close(self):
        self.ok = False
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception:
            pass


def haiku_stream(history, scenario, on_sentence, cancelled, model=CHAT_MODEL):
    """The writer streams the reply; each finished sentence goes to on_sentence() at once, so the voice can start
    before the reply is complete. -> (full text, error). Blocking: run it in a thread."""
    key = _read_key()
    if not key:
        return "", "no OPENROUTER_API_KEY (.env)"
    body = json.dumps({"model": model, "temperature": 0.6, "max_tokens": 320, "reasoning": {"enabled": False},
                       "stream": True,                      # 160 tokens cut Arabic replies mid-sentence
                       "messages": [{"role": "system", "content": CHAT_PROMPT.format(scenario=scenario)}]
                                   + _from_user(history[-CHAT_MAX_HISTORY:])}, ensure_ascii=False).encode("utf-8")
    full, pending = "", ""
    try:
        c = http.client.HTTPSConnection("openrouter.ai", timeout=15)
        c.request("POST", "/api/v1/chat/completions", body,
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "turn-taking demo"})
        r = c.getresponse()
        if r.status != 200:
            return "", f"LLM error {r.status}: {r.read()[:120]!r}"
        for line in r:
            if cancelled():
                return full, None
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            full += piece
            pending += piece
            m = None
            for m in _SENTENCE_END.finditer(pending):      # the last sentence end in what is pending
                pass
            # the first piece goes out at 2+ words («ألو، هلا والله،») so the voice starts early; later pieces at 4+
            if m and len(pending[:m.end()].split()) >= (4 if full != pending else 2):
                on_sentence(pending[:m.end()].replace("[END]", "").strip())
                pending = pending[m.end():]
        if pending.replace("[END]", "").strip():
            on_sentence(pending.replace("[END]", "").strip())
        return full.replace("[END]", "").strip(), None
    except Exception as ex:
        return full, f"LLM stream failed: {ex!r}"[:160]


def munsit_speak(text, voice_id, on_chunk, cancelled, speed=None):
    """Speak one piece with Munsit (Faseeh, native Saudi voices). The REST call returns the finished piece (no streaming:
    ~0.8-1.1 s for a short sentence, ~1.8 s for 7 s of speech), asked at 16 kHz mono so it plays as is. -> error or None."""
    key = _env("MUNSIT_API_KEY")
    if not key:
        return "no MUNSIT_API_KEY (.env)"
    body = {"voice_id": voice_id, "text": text, "sample_rate": ELEVEN_RATE}
    if speed:
        body["voice_settings"] = {"speed": speed}
    try:
        c = http.client.HTTPSConnection("api.munsit.com", timeout=20)
        c.request("POST", "/api/v1/text-to-speech/faseeh-v1-preview", json.dumps(body, ensure_ascii=False).encode(),
                  {"x-api-key": key, "Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            return f"Munsit error {r.status}: {data[:120]!r}"
        with wave.open(io.BytesIO(data)) as w:
            if w.getframerate() != ELEVEN_RATE or w.getnchannels() != 1:
                return f"Munsit sent {w.getframerate()} Hz x{w.getnchannels()}, expected {ELEVEN_RATE} Hz mono"
            pcm = w.readframes(w.getnframes())
        for i in range(0, len(pcm), 3200):
            if cancelled():
                return None
            on_chunk(pcm[i:i + 3200])
        return None
    except Exception as ex:
        return f"Munsit failed: {ex!r}"[:160]


def eleven_speak(text, voice_id, previous_text, on_chunk, cancelled, model=None, speed=None):
    """Speak one piece with ElevenLabs, streamed as pcm16 at 16 kHz. -> error or None."""
    if voice_id.startswith("munsit:"):                  # a Munsit voice shares the menu and the reply loop
        return munsit_speak(text, voice_id[7:], on_chunk, cancelled, speed)
    key = _env("ELEVENLABS_API_KEY")
    if not key:
        return "no ELEVENLABS_API_KEY (.env)"
    model = model or ELEVEN_MODEL
    body = {"text": text, "model_id": model, "language_code": "ar"}
    if speed:
        body["voice_settings"] = {"speed": speed}
    # previous_text keeps the intonation continuous across pieces, but the v3 models reject it with a 400 (only the first
    # piece of every reply was spoken, Nov 18): send it only to the older models, and retry once without it on a 400
    if previous_text and not model.startswith("eleven_v3"):
        body["previous_text"] = previous_text[-300:]
    try:
        c = http.client.HTTPSConnection("api.elevenlabs.io", timeout=15)
        c.request("POST", f"/v1/text-to-speech/{voice_id}/stream?output_format=pcm_{ELEVEN_RATE}",
                  json.dumps(body, ensure_ascii=False).encode(), {"xi-api-key": key, "Content-Type": "application/json"})
        r = c.getresponse()
        if r.status == 400 and "previous_text" in body:
            r.read()
            body.pop("previous_text")
            c = http.client.HTTPSConnection("api.elevenlabs.io", timeout=15)
            c.request("POST", f"/v1/text-to-speech/{voice_id}/stream?output_format=pcm_{ELEVEN_RATE}",
                      json.dumps(body, ensure_ascii=False).encode(), {"xi-api-key": key, "Content-Type": "application/json"})
            r = c.getresponse()
        if r.status != 200:
            return f"ElevenLabs error {r.status}: {r.read()[:120]!r}"
        carry = b""
        while True:
            if cancelled():
                return None
            b = r.read(3200)
            if not b:
                break
            pcm = carry + b
            cut = len(pcm) - len(pcm) % 2
            carry = pcm[cut:]
            if cut:
                on_chunk(pcm[:cut])
        return None
    except Exception as ex:
        return f"ElevenLabs failed: {ex!r}"[:160]


class Conversation:
    """Free-call mode: what was said so far, and the caller's audio for the turn in progress."""
    TURN_LOOKBACK_FRAMES = 150                          # an interruption started up to 3 s before the agent stopped

    def __init__(self, scenario, voice="shimmer", stt="local", writer="haiku"):
        self.scenario = scenario.strip() or DEFAULT_SCENARIO
        self.voice = voice
        self.writer = writer if writer in WRITERS else "haiku"
        self.stt = stt                                  # "local" (Whisper) or "cloud" (STT_MODEL) for the caller's turns
        self.speech_id = 0                              # bumped on every new line and on STOP: older speech is dropped
        self.history = []                               # OpenAI-style messages: agent = assistant, caller = user
        self.frames = collections.deque(maxlen=50 * 30)  # last 30 s of caller audio (20 ms frames)
        self.n = 0
        self.turn_start = 0
        self.agent_text = ""
        self.busy = False
        self.pending = None                             # a caller turn that ended while we were busy
        self.draft = None                               # an early reply draft for the current pause (see start_draft)

    def add_frame(self, caller):
        self.frames.append(caller.copy())
        self.n += 1

    def agent_ended(self, spoken_chars, stopped):
        """The agent's line ended (finished, or cut by our STOP). Mark the cut and start the caller's turn."""
        text = self.agent_text
        if stopped and spoken_chars is not None and spoken_chars < len(text) - 2:
            text = text[:spoken_chars].rstrip() + "…"
        if text:
            self.history.append({"role": "assistant", "content": text})
        self.agent_text = ""
        self.turn_start = self.n - (self.TURN_LOOKBACK_FRAMES if stopped else 25)

    def turn_audio(self):
        keep = min(len(self.frames), max(0, self.n - self.turn_start))
        if keep < 10:
            return None
        return np.concatenate(list(self.frames)[-keep:]).astype(np.float32) / 32768


class DemoSystem(MySystem):
    """The real system, plus a copy of every log line for the page."""

    def __init__(self, push, shadow=False):
        self._to_page = push                            # not "_push": MySystem._push stores the caller audio
        self._shadow = shadow                           # True: another system decides, we only show what we hear
        self.scribe_text = None                         # free call with live Scribe: a function giving the turn's text so far
        super().__init__()

    def _log(self, line):
        super()._log(line)
        self._to_page({"type": "log", "line": line, "shadow": self._shadow})

    def _eot_ready(self, smart):
        """Demo only: Whisper's text of a short pause is often garbled («ممكن يوم الاربع») and then looks finished.
        Live Scribe's text is accurate and keeps what a hesitating caller says: «آااه،», «يووم الأربعاء-». If it ends
        mid-thought, wait the longer time before taking the turn."""
        ready = super()._eot_ready(smart)
        if ready and self.scribe_text is not None:
            try:
                text = self.scribe_text().strip()
                incomplete = eot_incomplete(text)[0] or text.endswith("..")     # «كـ..» = a word cut off
            except Exception:
                incomplete = False
            if incomplete:
                return self.quiet_ms >= my_system.EOT_HOLD_MS
        return ready


SCRIBE_STOP = False     # set by --stop scribe: live Scribe's words can also trigger STOP (DEMO ONLY, not the scored system)

OTHER_SYSTEM = None     # set by --system: a MySystem class from another file that makes the decisions instead of ours
OTHER_NAME = ""


def load_other_system(path):
    """Import MySystem from another system file (e.g. a mentor's my_system_v1.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("other_system", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MySystem


async def handler(ws):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def push(msg):                                      # called from Whisper's worker thread too
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    system = DemoSystem(push, shadow=OTHER_SYSTEM is not None)   # ours: decides, or only shows what Whisper hears
    decider = OTHER_SYSTEM() if OTHER_SYSTEM is not None else system
    both = [system] if decider is system else [decider, system]
    push({"type": "info", "text": f"deciding system: {OTHER_NAME} (Whisper rows show what OUR system would do)"
          if decider is not system else "deciding system: ours (Whisper + Arabic rules)"})
    silent = np.zeros(320, dtype=np.int16)
    n = 0
    chat = None                                         # a Conversation in free-call mode, else None
    live = None                                         # ScribeLive stream for the caller's words (free call, "live")
    recording = []                                      # every mic frame of the current call, saved when it ends

    def save_recording():
        if len(recording) < 50:
            return
        REC_DIR.mkdir(exist_ok=True)
        path = REC_DIR / time.strftime("call-%H%M%S.wav")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(np.concatenate(recording).tobytes())
        push({"type": "info", "text": f"mic audio saved: {path.name} ({len(recording) * 20 / 1000:.0f} s)"})
        recording.clear()

    def transcribe_turn(audio, stt):
        """Blocking (never on the event loop): the caller's finished turn as text -> (text, source, notes).
        live Scribe falls back to batch Scribe; Scribe/cloud fall back to Whisper."""
        text, source, notes = "", "Whisper", []
        if stt == "live":
            if live is not None and live.ok:
                t0 = time.perf_counter()
                try:
                    text = asyncio.run_coroutine_threadsafe(live.commit(), loop).result(timeout=2.5)
                except Exception:
                    text = ""
                source = f"ElevenLabs live, committed in {(time.perf_counter() - t0) * 1000:.0f} ms"
            if not text:
                notes.append("live Scribe gave nothing (using batch Scribe instead)")
                stt = "scribe"
        if audio is not None and stt in ("scribe", "cloud"):
            fn, name = (scribe_transcribe, "ElevenLabs Scribe") if stt == "scribe" else (cloud_transcribe, STT_MODEL.split("/")[-1])
            text, ms, err = fn(audio)
            source = f"{name}, {ms:.0f} ms"
            if err:
                notes.append(err + " (using Whisper instead)")
                text, source = "", f"Whisper ({name} failed)"
        if not text and audio is not None and ASR.ok:
            text, _ = ASR.transcribe(audio, max_new_tokens=128)
        return clean_turn(text) or "…", source, notes

    def start_draft():
        """At 0.3 s of caller silence: transcribe the turn and let the writer start the reply in the background.
        Nothing is spoken and nothing enters the history; if the caller makes a sound again, it is thrown away."""
        d = {"eot_id": system._eot_id, "cancel": False, "used": False, "q": sentence_queue.Queue(), "res": {},
             "ready": threading.Event(), "t0": time.perf_counter()}
        audio, stt, hist = chat.turn_audio(), chat.stt, list(chat.history)
        scenario, model = chat.scenario, WRITERS.get(chat.writer, CHAT_MODEL)

        def run():
            text, source, notes = transcribe_turn(audio, stt)
            d.update(caller_text=text, source=source, notes=notes)
            d["ready"].set()
            if not d["cancel"]:
                d["res"]["text"], d["res"]["err"] = haiku_stream(hist + [{"role": "user", "content": text}], scenario,
                                                                 d["q"].put, lambda: d["cancel"], model)
            d["q"].put(None)

        threading.Thread(target=run, daemon=True).start()
        return d

    def eleven_reply(sid, draft=None):
        """Blocking (run in a thread): the writer streams sentences into a queue (or an early draft already did),
        ElevenLabs speaks them one by one."""
        q, res = (draft["q"], draft["res"]) if draft else (sentence_queue.Queue(), {})
        res["sounded"] = False

        def cancelled():
            return chat.speech_id != sid

        def produce():
            res["text"], res["err"] = haiku_stream(chat.history, chat.scenario, q.put, cancelled,
                                                   WRITERS.get(chat.writer, CHAT_MODEL))
            q.put(None)

        def on_chunk(pcm):
            if not cancelled():
                if not res["sounded"]:
                    res["first_sound"] = time.perf_counter()
                res["sounded"] = True
                push(pcm)

        if draft is None:
            threading.Thread(target=produce, daemon=True).start()
        spoken = []
        while True:
            s = q.get()
            if s is None:
                break
            if cancelled() or res.get("tts_err"):
                continue
            spoken.append(s)
            chat.agent_text = " ".join(spoken)
            push({"type": "agent_text", "speech": sid, "text": chat.agent_text})
            err = eleven_speak(s, ELEVEN_VOICES[chat.voice], " ".join(spoken[:-1]), on_chunk, cancelled,
                               *ELEVEN_STYLE.get(chat.voice, (None, None)))
            if err:
                res["tts_err"] = err
                push({"type": "info", "text": f"voice failed on «{s}»: {err}"})   # never go silent without a trace
        return res

    async def agent_speaks(caller_text=None):
        """Free-call mode: transcribe the caller's turn (if any), ask the LLM, send the agent's line to the page."""
        if chat is None:
            return
        if chat.busy:                                   # still preparing or speaking: answer this turn right after
            chat.pending = chat.pending or caller_text or "turn"
            return
        chat.busy = True
        push({"type": "thinking", "on": True})
        t_take, quiet_at_take = time.perf_counter(), system.quiet_ms
        d = chat.draft                                  # an early draft for exactly this pause?
        use = (caller_text is None and d is not None and not d["cancel"] and d["eot_id"] == system._eot_id
               and chat.voice in ELEVEN_VOICES and WRITER == "haiku")
        if d is not None and not use:
            d["cancel"] = True
        if d is not None:
            d["used"] = use
        chat.draft = None
        try:
            if caller_text is None:
                if use:
                    await loop.run_in_executor(None, d["ready"].wait, 10)
                    caller_text = d.get("caller_text", "…")
                    source, notes = d.get("source", "Whisper") + " · drafted early", d.get("notes", [])
                else:
                    caller_text, source, notes = await loop.run_in_executor(None, transcribe_turn, chat.turn_audio(), chat.stt)
                for n in notes:
                    push({"type": "info", "text": n})
                push({"type": "caller_turn", "text": caller_text, "source": source})
                if live is not None:
                    live.new_turn()                     # the agent takes the floor: next words = the next turn
            chat.history.append({"role": "user", "content": caller_text})
            chat.speech_id += 1
            sid = chat.speech_id
            chat.agent_text = ""
            written = None
            if chat.voice in ELEVEN_VOICES and WRITER == "haiku":
                # fast path: Haiku streams sentences, ElevenLabs speaks each one as soon as it is written
                push({"type": "agent_say", "text": "", "audio": "stream", "rate": ELEVEN_RATE, "speech": sid})
                res = await loop.run_in_executor(None, eleven_reply, sid, d if use else None)
                if res.get("first_sound"):
                    after = (res["first_sound"] - t_take) * 1000
                    push({"type": "info", "text": f"reply timing: you were quiet {quiet_at_take} ms before TAKE_TURN, "
                                                  f"first sound {after:.0f} ms after it = {quiet_at_take + after:.0f} ms "
                                                  f"of silence{' (reply drafted early)' if use else ''}"})
                if res.get("err") and not res.get("text"):
                    push({"type": "info", "text": res["err"] + " (the GPT voice writes instead)"})
                elif res.get("tts_err") and not res.get("sounded"):
                    push({"type": "info", "text": res["tts_err"] + " (the GPT voice reads instead)"})
                    written = res["text"]
                    chat.speech_id += 1                 # a new speech at the GPT voice's own sample rate
                    sid = chat.speech_id
                else:
                    push({"type": "agent_audio_end", "speech": sid, "failed": False, "text": chat.agent_text,
                          "end": bool(_GOODBYE.search(chat.agent_text))})
                    return
            if written is None and WRITER == "haiku":   # step 1: Haiku writes the line
                written, err = await loop.run_in_executor(None, chat_reply, chat.history, chat.scenario)
                if err or not written:
                    push({"type": "info", "text": (err or "empty reply") + " (the voice model writes instead)"})
                    written = None
                elif chat.speech_id != sid:             # the caller spoke meanwhile: drop this line
                    return
                else:
                    written = written.replace("[END]", "").strip()
            push({"type": "agent_say", "text": written or "", "audio": "stream", "rate": TTS_RATE, "speech": sid})
            gpt_voice = chat.voice if chat.voice in TTS_VOICES else TTS_VOICES[0]

            def on_text(said):                          # the transcript of what the voice is saying, as it grows
                if chat.speech_id == sid:
                    chat.agent_text = said
                    push({"type": "agent_text", "speech": sid, "text": said})

            if written:                                 # step 2: the voice only READS Haiku's line
                messages = read_messages(written)
            else:                                       # the voice model writes and speaks in one call
                messages = ([{"role": "system", "content": CHAT_PROMPT.format(scenario=chat.scenario) + VOICE_PROMPT_EXTRA}]
                            + _from_user(chat.history[-CHAT_MAX_HISTORY:]))
            said, err = await loop.run_in_executor(None, voice_stream, messages, gpt_voice,
                                                   lambda pcm: push(pcm) if chat.speech_id == sid else None,
                                                   on_text, lambda: chat.speech_id != sid)
            if written and said and SequenceMatcher(None, normalize(written), normalize(said)).ratio() < 0.85:
                push({"type": "info", "text": f"the voice changed the words: wrote «{written}» / said «{said}»"})
            failed = bool(err) and not said
            if failed and written:                      # no voice at all: the browser speaks Haiku's line
                push({"type": "info", "text": err + " (falling back to the browser voice)"})
                chat.agent_text = written
                failed = True
            elif failed:                                # no voice at all: Haiku writes, the browser speaks
                push({"type": "info", "text": err + " (falling back to Haiku + the browser voice)"})
                reply, err2 = await loop.run_in_executor(None, chat_reply, chat.history, chat.scenario)
                chat.agent_text = (reply or "").replace("[END]", "").strip()
                if err2:
                    push({"type": "info", "text": err2})
            push({"type": "agent_audio_end", "speech": sid, "failed": failed, "text": chat.agent_text,
                  "end": bool(_GOODBYE.search(chat.agent_text))})
        finally:
            chat.busy = False
            push({"type": "thinking", "on": False})
            held, chat.pending = chat.pending, None
            if held == "turn" and not decider.agent_talking:  # the caller finished a turn while we were busy and
                asyncio.create_task(agent_speaks())         # the agent is not speaking over it: answer it now

    # --- "Scribe stop" mode (demo only): live Scribe reads the caller's words while the agent talks, far more accurately
    # than Whisper on a hard voice, and the same rules (turn_rules.decide, with the same loudness gates) judge them.
    # Whisper keeps running in parallel: the first of the two to say STOP wins. Needs headphones (or good echo
    # cancellation): Scribe is not muted while the agent talks.
    ss = {"mark": None, "peak": -99.0, "last": "", "t0": 0.0}

    def scribe_log(text, stop, why, peak, ref):
        ms = (time.perf_counter() - ss["t0"]) * 1000
        rel = f"{peak - ref:+.0f}" if ref is not None else "n/a"
        system._log(f"[asr] burst {system._burst_id} {ms / 1000:.1f}s ({ms:.0f} ms, Scribe live, pk {peak:.0f} dB, "
                    f"vs caller {rel} dB): {text!r} -> {'STOP' if stop else 'keep'} (Scribe: {why})")

    def scribe_judge(text, peak):
        ref = system._burst_ref
        stop, why = decide(text, " ".join(system.agent_hist), peak, ref)
        scribe_log(text, stop, why, peak, ref)
        if stop and system.agent_talking and not system.stopped:
            system._stop_flag = True                    # the next frame returns STOP, as for Whisper's own decision
        return stop

    async def scribe_burst_over(mark, peak):
        """The burst ended and nothing stopped the agent: final text from Scribe, one last look, then forget the words
        (a kept «أيوه» must not become the start of the caller's next turn)."""
        await live.commit(timeout=1.0)
        if system.stopped:
            return
        if not system.agent_talking:
            # the agent finished while the caller was still talking or just after (an answer to its last words, «إيه
            # أكيد» to «ممكن تنتظر؟»): it is the caller's turn, keep the words and let TAKE_TURN follow their silence
            if " ".join(live.turn[mark:]).strip():
                system.caller_spoke, system.quiet_ms = True, 0
            return
        text = " ".join(live.turn[mark:]).strip()
        if text and ss["mark"] is None and scribe_judge(text, peak):
            return
        if ss["mark"] is None:
            if text:
                ss["kept"] = (text, time.perf_counter())    # if the agent ends within 1.5 s, this was its answer
            del live.turn[mark:]
            live.partial = ""

    def scribe_stop_step(caller):
        talking = system.agent_talking and not system.stopped
        if talking and system._in_burst:
            if ss["mark"] is None:
                ss.update(mark=len(live.turn), peak=-99.0, last="", t0=time.perf_counter())
            ss["peak"] = max(ss["peak"], float(loudness_db(caller)))
            text = " ".join(live.turn[ss["mark"]:] + ([live.partial] if live.partial else [])).strip()
            if text and text != ss["last"]:
                ss["last"] = text
                scribe_judge(text, ss["peak"])
        elif ss["mark"] is not None:
            mark, peak, ss["mark"] = ss["mark"], ss["peak"], None
            if system.agent_talking and not system.stopped and not system._stop_flag:
                asyncio.create_task(scribe_burst_over(mark, peak))

    async def sender():
        while True:
            item = await queue.get()
            await ws.send(item if isinstance(item, bytes) else json.dumps(item, ensure_ascii=False))   # bytes = agent voice

    task = asyncio.create_task(sender())
    try:
        async for raw in ws:
            if isinstance(raw, bytes):                  # 640 bytes: 320 caller samples, int16
                caller = np.frombuffer(raw, dtype="<i2")
                if len(caller) != 320:
                    continue
                if chat is not None:
                    chat.add_frame(caller)
                if live is not None:
                    hold = bool(getattr(decider, "agent_talking", False)) and not getattr(decider, "stopped", False)
                    # after a STOP: the burst that caused it, even if it already ended (Whisper's last look comes after
                    # the burst; «دقيقة معلش، خلّي يوم الثلاثاء» was over before the STOP and was lost, Nov 19)
                    stopped_now = getattr(decider, "stopped", False)
                    burst = (system._n - system._burst_start) // 320 if (system._in_burst or stopped_now) else 0
                    if SCRIBE_STOP and decider is system:
                        live.feed(caller)               # never muted: its words can stop the agent
                        scribe_stop_step(caller)
                    else:
                        live.feed(caller, hold, burst)
                if len(recording) < 50 * 600:           # at most 10 minutes per call
                    recording.append(caller.copy())
                if decider is not system:
                    system.on_frame(silent, caller)     # shadow: Whisper reads and logs, its decisions are ignored
                out = decider.on_frame(silent, caller)
                for d in ([out] if isinstance(out, (str, dict)) else out or []):
                    d = d if isinstance(d, str) else d.get("decision")
                    push({"type": "decision", "decision": d})
                    if d == "TAKE_TURN" and chat is not None:
                        asyncio.create_task(agent_speaks())
                if chat is not None and chat.voice in ELEVEN_VOICES and WRITER == "haiku":
                    dr = chat.draft
                    if dr is not None and not dr["used"] and dr["eot_id"] != system._eot_id:
                        dr["cancel"] = True             # the caller made a sound again: that draft is for nothing
                        chat.draft = None
                    if (chat.draft is None and not chat.busy and not decider.agent_talking and system.caller_spoke
                            and system.quiet_ms >= DRAFT_AFTER_MS):
                        chat.draft = start_draft()
                n += 1
                if n % 5 == 0:                          # 10 level updates a second for the meter
                    if decider is system:
                        ref = system._ref_db if system._ref_db > LOUD_DB else None
                        floor = float(np.percentile(system.hist, FLOOR_PERCENTILE)) if system.hist else None
                    else:                               # the other system's own voice level and noise floor, if it has them
                        ref, floor = getattr(decider, "ref_db", None), getattr(decider, "noise_db", None)
                    push({"type": "level", "db": round(float(loudness_db(caller)), 1),
                          "floor": None if floor is None else round(floor, 1),
                          "ref": None if ref is None else round(ref, 1)})
                continue
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "call_end":
                save_recording()
                system.scribe_text = None
                if live is not None:                    # stop streaming (and paying) when the call ends
                    await live.close()
                    live = None
            elif t == "call_start":
                save_recording()
                chat = None
                if live is not None:
                    await live.close()
                    live = None
                for s in both:
                    s.on_scenario_start({"use_case": msg.get("use_case", ""), "language": "ar"})
                push({"type": "info", "text": f"call started: {msg.get('use_case', '')}"})
            elif t == "chat_start":                     # free call: the agent opens with an LLM line
                save_recording()
                system.scribe_text = None
                chat = Conversation(msg.get("scenario", ""), msg.get("voice", "shimmer"), msg.get("stt", "local"),
                                    msg.get("writer", "haiku"))
                if live is not None:
                    await live.close()
                    live = None
                if chat.stt == "live":
                    live = ScribeLive(lambda text: push({"type": "live_partial", "text": text}))
                    err = await live.start()
                    if err:
                        push({"type": "info", "text": err + " (using batch Scribe instead)"})
                        live, chat.stt = None, "scribe"
                    else:
                        lv = live                       # the end-of-turn check reads this call's live text
                        system.scribe_text = lambda: " ".join(lv.turn + ([lv.partial] if lv.partial else []))
                for s in both:
                    s.on_scenario_start({"use_case": "free call", "language": "ar"})
                push({"type": "info", "text": "free call started: " + chat.scenario})
                if SCRIBE_STOP:
                    push({"type": "info", "text": "SCRIBE STOP MODE (demo only): live Scribe's words can also stop the agent; "
                          "the scored system decides with Whisper alone" if live is not None else
                          "Scribe stop mode needs 'Your words: live' (it is off for this call)"})
                asyncio.create_task(agent_speaks("ألو؟"))       # the customer picks up
            elif t == "chat_silence" and chat is not None:   # the page's timer: no answer since the agent's line
                # only if the caller really stayed silent: the timer counts from the agent's last word and cannot know
                # the caller is mid-sentence; it made the agent talk over a long turn twice (Nov 18)
                if not system.caller_spoke and not system._in_burst:
                    asyncio.create_task(agent_speaks("…"))     # silence: the agent checks in
            elif t == "agent_start":
                ss["mark"] = None
                ss.pop("kept", None)
                for s in both:
                    s.on_agent_start(msg.get("text", ""), int(msg.get("duration_ms", 0)))
            elif t == "agent_end":
                stopped = getattr(decider, "stopped", False)
                for s in both:
                    s.on_agent_end()
                kept = ss.pop("kept", None)
                if SCRIBE_STOP and live is not None and not stopped and kept and time.perf_counter() - kept[1] < 1.5:
                    live.turn[:] = [kept[0]]            # said over the agent's last words: that is the caller's answer
                    system.caller_spoke = True
                    system.quiet_ms = int((time.perf_counter() - kept[1]) * 1000)
                    push({"type": "info", "text": f"you answered over the agent's last words: «{kept[0]}» counts as your turn"})
                if live is not None and chat is not None and chat.agent_text:
                    live.agent_said(chat.agent_text)    # its echo will be removed from the caller's next turn
                if chat is not None:
                    if stopped:
                        chat.speech_id += 1             # STOP: the rest of the speech is cancelled (and not paid for)
                    chat.agent_ended(msg.get("spoken_chars"), stopped)
    finally:
        save_recording()
        task.cancel()
        if live is not None:
            await live.close()
        system.agent_talking = system.caller_spoke = False   # an old page's worker stops its GPU warm-up decodes


async def main(port):
    ok = "Whisper ready" if ASR.ok else "Whisper NOT available: energy rule only"
    async with serve(handler, "127.0.0.1", port, process_request=process_request, max_size=2 ** 20, ping_interval=None):
        print(f"[demo] {ok}. Open http://127.0.0.1:{port} in Chrome or Edge (headphones recommended).", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--stop", choices=["whisper", "scribe"], default="whisper",
                    help="scribe = demo only: live Scribe's words can also stop the agent (use headphones)")
    ap.add_argument("--system", default="", help="another system file decides instead of ours; 'v1' = my_system_v1.py")
    a = ap.parse_args()
    if a.system:
        path = HERE / ("my_system_v1.py" if a.system == "v1" else a.system)
        OTHER_SYSTEM, OTHER_NAME = load_other_system(path), path.name
        print(f"[demo] decisions come from {path.name}; our system only shows what Whisper hears", flush=True)
    SCRIBE_STOP = a.stop == "scribe"
    if SCRIBE_STOP:
        print("[demo] SCRIBE STOP MODE: live Scribe can also stop the agent (demo only; use headphones)", flush=True)
    my_system.LOG_FILE = "demo_log.txt"                 # keep the benchmark's asr_log.txt clean
    ASR.load()
    asyncio.run(main(a.port))
