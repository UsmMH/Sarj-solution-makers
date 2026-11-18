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
import re
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
from turn_rules import normalize
from my_system import ASR, FLOOR_PERCENTILE, LOUD_DB, MySystem, loudness_db
from ttbench.dataset import read_wav_stereo

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
- عبارات طبيعية لما تناسب: أبشر، ولا يهمك، تمام، طال عمرك، الله يعافيك، على راحتك.
- ردود قصيرة: جملة أو جملتين، أقل من 20 كلمة. حروف عربية فقط، بدون حروف لاتينية أو رموز أو أقواس.
- كلام العميل جاي من نظام تعرّف صوت وممكن يكون فيه أخطاء؛ إذا ما فهمتي، اطلبي منه يعيد بلطف.
- إذا جملتك السابقة تنتهي بـ «…» يعني العميل قاطعك هنا: ردي على كلامه، لا تعيدين جملتك.
- إذا كلام العميل «…» يعني سكت وما رد: اسأليه بلطف إذا يسمعك.
- قولي دورك فقط ثم وقفي، لا تجاوبين عن العميل ولا تخترعين أسماء.
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


class Conversation:
    """Free-call mode: what was said so far, and the caller's audio for the turn in progress."""
    TURN_LOOKBACK_FRAMES = 150                          # an interruption started up to 3 s before the agent stopped

    def __init__(self, scenario, voice="shimmer"):
        self.scenario = scenario.strip() or DEFAULT_SCENARIO
        self.voice = voice
        self.speech_id = 0                              # bumped on every new line and on STOP: older speech is dropped
        self.history = []                               # OpenAI-style messages: agent = assistant, caller = user
        self.frames = collections.deque(maxlen=50 * 30)  # last 30 s of caller audio (20 ms frames)
        self.n = 0
        self.turn_start = 0
        self.agent_text = ""
        self.busy = False
        self.pending = None                             # a caller turn that ended while we were busy

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
        super().__init__()

    def _log(self, line):
        super()._log(line)
        self._to_page({"type": "log", "line": line, "shadow": self._shadow})


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

    async def agent_speaks(caller_text=None):
        """Free-call mode: transcribe the caller's turn (if any), ask the LLM, send the agent's line to the page."""
        if chat is None:
            return
        if chat.busy:                                   # still preparing or speaking: answer this turn right after
            chat.pending = chat.pending or caller_text or "turn"
            return
        chat.busy = True
        push({"type": "thinking", "on": True})
        try:
            if caller_text is None:
                audio = chat.turn_audio()
                text = ""
                if audio is not None and ASR.ok:
                    text, _ = await loop.run_in_executor(None, lambda: ASR.transcribe(audio, max_new_tokens=128))
                caller_text = clean_turn(text) or "…"
                push({"type": "caller_turn", "text": caller_text})
            chat.history.append({"role": "user", "content": caller_text})
            chat.speech_id += 1
            sid = chat.speech_id
            chat.agent_text = ""
            written = None
            if WRITER == "haiku":                       # step 1: Haiku writes the line
                written, err = await loop.run_in_executor(None, chat_reply, chat.history, chat.scenario)
                if err or not written:
                    push({"type": "info", "text": (err or "empty reply") + " (the voice model writes instead)"})
                    written = None
                elif chat.speech_id != sid:             # the caller spoke meanwhile: drop this line
                    return
                else:
                    written = written.replace("[END]", "").strip()
            push({"type": "agent_say", "text": written or "", "audio": "stream", "rate": TTS_RATE, "speech": sid})

            def on_text(said):                          # the transcript of what the voice is saying, as it grows
                if chat.speech_id == sid:
                    chat.agent_text = said
                    push({"type": "agent_text", "speech": sid, "text": said})

            if written:                                 # step 2: the voice only READS Haiku's line
                messages = read_messages(written)
            else:                                       # the voice model writes and speaks in one call
                messages = ([{"role": "system", "content": CHAT_PROMPT.format(scenario=chat.scenario) + VOICE_PROMPT_EXTRA}]
                            + _from_user(chat.history[-CHAT_MAX_HISTORY:]))
            said, err = await loop.run_in_executor(None, voice_stream, messages, chat.voice,
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
        if held == "turn" and not decider.agent_talking:  # the caller finished a turn while we were busy and the
            asyncio.create_task(agent_speaks())         # agent is not speaking over it: answer that turn now

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
            elif t == "call_start":
                save_recording()
                chat = None
                for s in both:
                    s.on_scenario_start({"use_case": msg.get("use_case", ""), "language": "ar"})
                push({"type": "info", "text": f"call started: {msg.get('use_case', '')}"})
            elif t == "chat_start":                     # free call: the agent opens with an LLM line
                save_recording()
                chat = Conversation(msg.get("scenario", ""), msg.get("voice", "shimmer"))
                for s in both:
                    s.on_scenario_start({"use_case": "free call", "language": "ar"})
                push({"type": "info", "text": "free call started: " + chat.scenario})
                asyncio.create_task(agent_speaks("ألو؟"))       # the customer picks up
            elif t == "chat_silence" and chat is not None:   # the caller said nothing: the agent goes on
                asyncio.create_task(agent_speaks("…"))         # silence: the agent checks in
            elif t == "agent_start":
                for s in both:
                    s.on_agent_start(msg.get("text", ""), int(msg.get("duration_ms", 0)))
            elif t == "agent_end":
                stopped = getattr(decider, "stopped", False)
                for s in both:
                    s.on_agent_end()
                if chat is not None:
                    if stopped:
                        chat.speech_id += 1             # STOP: the rest of the speech is cancelled (and not paid for)
                    chat.agent_ended(msg.get("spoken_chars"), stopped)
    finally:
        save_recording()
        task.cancel()
        system.agent_talking = system.caller_spoke = False   # an old page's worker stops its GPU warm-up decodes


async def main(port):
    ok = "Whisper ready" if ASR.ok else "Whisper NOT available: energy rule only"
    async with serve(handler, "127.0.0.1", port, process_request=process_request, max_size=2 ** 20, ping_interval=None):
        print(f"[demo] {ok}. Open http://127.0.0.1:{port} in Chrome or Edge (headphones recommended).", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--system", default="", help="another system file decides instead of ours; 'v1' = my_system_v1.py")
    a = ap.parse_args()
    if a.system:
        path = HERE / ("my_system_v1.py" if a.system == "v1" else a.system)
        OTHER_SYSTEM, OTHER_NAME = load_other_system(path), path.name
        print(f"[demo] decisions come from {path.name}; our system only shows what Whisper hears", flush=True)
    my_system.LOG_FILE = "demo_log.txt"                 # keep the benchmark's asr_log.txt clean
    ASR.load()
    asyncio.run(main(a.port))
