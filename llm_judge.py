"""Second opinion from a cloud LLM (through OpenRouter) for bursts the text rules cannot settle.

turn_rules.decide() is sure about clear cues («لحظة», «استنى»), echoes, Whisper's stock phrases and plain
acknowledgements. In between ("too little to tell", "a question, but quiet") it says keep. Only those in-between
texts come here: a fast LLM reads the agent's sentence and the caller's words and answers STOP or KEEP.

Never on the critical path: the judge has its own thread, keeps only the newest request, caches texts it has
already judged, and gives up after LLM_TIMEOUT_MS. No answer = the rules' keep stands.

    $env:LLM = "off"                        # rules only, as before
    $env:LLM_MODEL = "google/gemini-2.5-flash"
    $env:LLM_QUIET = "on"                   # also ask about quiet bursts (blocked by the -20 dB gate today)

The key is read from OPENROUTER_API_KEY: this window's variable, the git-ignored .env file, or `setx`.
"""
import http.client
import json
import os
import threading
import time

LLM_ON = os.environ.get("LLM", "on").lower() != "off"
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-5.5")   # fewest false stops in llm_replay.py (Nov 17)
LLM_TIMEOUT_MS = int(os.environ.get("LLM_TIMEOUT_MS", "1200"))   # past ~1.5 s after "evident" a stop earns nothing
LLM_QUIET = os.environ.get("LLM_QUIET", "off").lower() == "on"
HOST = "openrouter.ai"
PATH = "/api/v1/chat/completions"
IDLE_RECONNECT_S = 20                                              # reopen the TLS connection before the server drops it

SYSTEM_PROMPT = """You are part of a voice agent on an Arabic (mostly Saudi dialect) phone call.
The AGENT is speaking right now. Meanwhile the caller's microphone picked up some sound, and a speech recogniser
turned it into text. The recogniser is often wrong on the first words, and on noise or TV it invents words.

Decide whether the agent must stop talking:
STOP = the words clearly show the caller is taking the floor: interrupting, objecting, refusing, correcting,
asking a question, asking the agent to wait, or saying they don't understand.
KEEP = everything else: a backchannel or agreement (ايه، تمام، اها، صح), hesitation or filler, laughter, talking to
someone else in the room, TV or background speech, the agent's own words echoing back, and noise.
A single word, a name, or a fragment with no clear request, objection or question is KEEP: the recogniser makes
such fragments out of TV, noise and the agent's own voice all the time. Stopping wrongly is costly; when in doubt,
KEEP.

Answer with exactly one word: STOP or KEEP."""


def _read_key():
    """This window's variable, else the .env file next to this script (git-ignored), else the setx user variable."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), encoding="utf-8-sig") as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name.strip() == "OPENROUTER_API_KEY" and value.strip():
                    return value.strip().strip("'\"")
    except OSError:
        pass
    if os.name != "nt":
        return ""
    try:                                                           # setx saves here; already-open windows don't see it
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, "OPENROUTER_API_KEY")[0]).strip()
    except OSError:
        return ""


class LlmJudge:
    def __init__(self, model=LLM_MODEL, timeout_ms=LLM_TIMEOUT_MS):
        self.model = model
        self.timeout_s = timeout_ms / 1000
        self.key = _read_key() if LLM_ON else ""
        self.ok = bool(self.key)
        self._conn = None
        self._last_use = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None                                           # newest request only
        self._cache = {}                                           # (agent, text) -> verdict
        self._fails = 0

    # --- one blocking call (used by the thread, and by the offline replay) ---------------------------------
    def ask(self, text, agent_text=""):
        """-> ("STOP" | "KEEP" | None, ms). None = no usable answer (error, timeout, odd reply)."""
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Agent is saying: «{agent_text or '(unknown)'}»\n"
                                            f"Caller channel (speech recogniser): «{text}»"},
            ],
            "max_tokens": 16,                                     # some models spend a few tokens before the answer
            "temperature": 0,
            "reasoning": {"enabled": False},                       # no hidden thinking: it ate the whole budget
            "provider": {"sort": "latency"},
        }).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "turn-taking"}
        t0 = time.perf_counter()
        for attempt in (1, 2):                                     # a dropped keep-alive connection gets one retry
            try:
                conn = self._connection()
                conn.request("POST", PATH, body, headers)
                resp = conn.getresponse()
                raw = resp.read()
                self._last_use = time.monotonic()
                ms = (time.perf_counter() - t0) * 1000
                if resp.status != 200:
                    self._fails += 1
                    return None, ms
                out = json.loads(raw)["choices"][0]["message"].get("content") or ""
                word = out.strip().upper()
                self._fails = 0
                if word.startswith("STOP"):
                    return "STOP", ms
                if word.startswith("KEEP"):
                    return "KEEP", ms
                return None, ms
            except (http.client.HTTPException, OSError, ValueError, KeyError, IndexError):
                self._close()
                if attempt == 2 or (time.perf_counter() - t0) > self.timeout_s:
                    self._fails += 1
                    return None, (time.perf_counter() - t0) * 1000
        return None, (time.perf_counter() - t0) * 1000

    def _connection(self):
        if self._conn is not None and time.monotonic() - self._last_use > IDLE_RECONNECT_S:
            self._close()
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(HOST, timeout=self.timeout_s)
        return self._conn

    def _close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def warm(self):
        """One real request at start-up: opens the connection and shows the key works, before any call."""
        verdict, ms = self.ask("ايه", "معك ريم من المتجر")
        return verdict is not None, ms

    # --- background use from my_system -----------------------------------------------------------------------
    def start(self, log):
        self._log = log
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text, agent_text, on_stop):
        """Ask about this text later, on the judge thread. on_stop() is called if the answer is STOP."""
        key = (agent_text, text)
        verdict = self._cache.get(key)
        if verdict is not None:
            if verdict == "STOP":
                on_stop()
            return
        with self._lock:
            self._job = (text, agent_text, on_stop)
        self._wake.set()

    def _run(self):
        while True:
            self._wake.wait(timeout=IDLE_RECONNECT_S / 2)
            with self._lock:
                job, self._job = self._job, None
                self._wake.clear()
            if job is None:
                if self._conn is None or time.monotonic() - self._last_use > IDLE_RECONNECT_S:
                    try:                                           # keep a fresh TLS connection ready
                        self._close()
                        self._connection().connect()
                        self._last_use = time.monotonic()
                    except OSError:
                        self._close()
                continue
            text, agent_text, on_stop = job
            if (agent_text, text) in self._cache:
                continue
            verdict, ms = self.ask(text, agent_text)
            if verdict is not None:
                self._cache[(agent_text, text)] = verdict
            self._log(f"[llm] ({ms:.0f} ms): {text!r} -> {verdict or 'no answer'}")
            if verdict == "STOP" and ms <= self.timeout_s * 1000:
                on_stop()
