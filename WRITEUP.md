# Write-up: Team 4 (for now :D)

## 1. The idea

**Approach, in a few words:** hybrid: acoustic gating + local speech recognition + Arabic rules that read both the
caller's words and how loud they are compared with the caller's own voice.

An adaptive energy detector notices that the caller is making sound while the agent talks. A local Whisper model
(large-v3-turbo on the laptop GPU) re-reads that burst every ~300 ms, and Saudi/Gulf Arabic rules decide: a floor bid
(«لحظة», «لا لا», «اصبر», «ماني فاهم», a question, a real sentence) means STOP; a backchannel («إيه», «تمام»),
an echo of the agent, side talk, TV or noise means keep talking. Unlike the baseline, which only hears energy, we
understand the words, and we judge loudness *relative to this caller*, so a soft speaker or a quiet line is not
mistaken for background noise. For end of turn, we read the caller's last words and wait longer when the sentence
sounds unfinished («و…», «يعني», a filler).

## 2. Why this approach

The baseline (B1, 41.0) stops on any loud sound, so backchannels, coughs and TV all stop the agent. A loudness-only
rule of ours reached 86.1 on the dev calls but cannot tell «إيه» from «لحظة»: in Arabic phone calls the difference
between "keep going" and "stop" is in the words, and backchannels are very frequent. So we added recognition, then
spent most of our time on the places where it fails.

What we found and changed, each time measured in real-time runs:

- **Whisper's "subtitle credit".** On short, abrupt clips of real speech Whisper often outputs
  «ترجمة نانسي قنقر» ("translated by …"), a phrase from its training subtitles. It cost up to a second on
  interruptions. Giving it 400 ms of audio before the burst removed it from all 27 short stop clips we tested (was
  6/27), and our clean score went from 86.5–90.1 to **94.6**.
- **A fixed loudness gate breaks on quiet callers.** We first blocked weak evidence below −20 dBFS (it held on the
  dev calls). A copy of the 24 calls with the caller 12 dB quieter dropped us to 74.4. We now compare each burst with
  the caller's own voice earlier in the call (only sustained speech counts, so a mic bump cannot set it): 93.5 on the
  same quiet copy.
- **Live testing found what the dev calls could not:** «اصبر» ("wait") was missing from our cues; after a STOP on a
  short «لا» the agent stayed silent for 8 s; a headset noise gate chopped words.

What did not work (tested, then dropped): a cloud LLM as STOP/KEEP judge for unclear text (2–4 false stops per model
on our logged decodes, almost no gain, 0.5–1 s per answer); cloud speech recognisers (best 0.83 match with the true
text in 2.4–3.9 s, against Whisper's 0.75 in 0.5 s); a Whisper text prompt (Whisper read the prompt back as speech,
causing a false STOP); a beam-5 retry (it swapped one invented phrase for another); a looser two-word rule (it fixes
one late stop but false-stops on TV speech).

## 3. How it works

```
caller audio, 20 ms frames (agent audio unused; agent TEXT used for echo detection)
 │
 ├─ adaptive noise floor (20th percentile of last 3 s + 10 dB) ── burst of caller sound? (only while the agent talks)
 │
 ├─ Whisper worker thread (never blocks the 20 ms loop): decodes the growing burst + 400 ms lead-in,
 │   first after 300 ms, then every ~100 ms of new audio, newest request only
 │
 ├─ decide(text, last 3 agent sentences, burst peak, caller reference voice):
 │     keep:  nothing heard · Whisper stock phrases · echo of the agent · talk to someone in the room
 │            · burst more than 12 dB quieter than the caller's voice (TV, speakerphone echo)
 │     STOP:  cue word (لحظة، استنى، اسمع، بس، اصبر، «لا» at the start…) or one of 60 phrases
 │            (ما فهمت، وش تقول، ما يناسبني، غلط، عندي سؤال…)
 │     STOP:  a question, or 3+ real words, unless more than 6 dB quieter than the caller's voice
 │
 └─ end of turn (agent silent): caller quiet 300 ms → Whisper reads their last 4 s → unfinished? wait up to 1.6 s,
    else 0.7 s → TAKE_TURN. After our own STOP the caller has the floor, so TAKE_TURN follows their silence.
Fallback: if Whisper fails 3 times, a loudness-only rule decides.
```

The caller reference voice is the loudest frame of sustained caller speech so far in the call (at least half of a
200 ms window above the noise floor), taken outside the burst being judged and reset for every call. Libraries:
faster-whisper (CTranslate2), NumPy, websockets. Model: `mobiuslabsgmbh/faster-whisper-large-v3-turbo`.

A live browser demo (`demo.py`) runs the same system on a microphone, with either the real agent sentences of the dev
calls or a free conversation with a voice agent. Around the scored engine it adds three cloud layers, demo only:
ElevenLabs real-time transcription of the caller's words, Claude Haiku writing short Saudi-style replies sentence by
sentence, and a Saudi voice (ElevenLabs Voice Library or Munsit) speaking each sentence as it arrives. STOP cuts the
voice at once; the agent answers about 2.0-2.5 s after the caller goes quiet. Two demo-only extras, both labelled on the
page: an optional **Scribe stop mode** (`--stop scribe`), where the live transcript can also trigger STOP through the
same rules and loudness gates (it reads the first word of a hard voice better than Whisper), and a longer wait before
taking the turn (1.5 s, 3.2 s when the live text ends mid-word) because a person talking freely pauses to think. The
scored system decides with local Whisper alone; none of this is part of it.

## 4. Speed and cost

| | |
|---|---|
| Median stop latency on the dev set (real-time run, from the report) | 416 ms after the intent was evident (p90 754 ms), final commit `d12ca5b`. Through an internet tunnel (practice checkpoint) the harness round trip was ~187 ms, which adds about half of that to every decision. |
| Where that time goes (each stage, roughly) | 300 ms of burst before the first decode; Whisper decode median ~280 ms; up to 100 ms waiting for the next decode slot; harness round trip ~1 ms on a local network |
| Hardware it ran on, and how busy it was | Windows laptop, NVIDIA RTX 3060 Laptop GPU (6 GB), ~2.4 GB GPU memory used; must run on mains power (on battery, decodes are ~35 % slower) |
| Model sizes | Whisper large-v3-turbo, ~809 M parameters, ~1.6 GB in float16 |
| Cloud calls per minute of audio, and their typical delay | 0 (fully local) |
| What it would cost to run on a real call | One mid-range GPU serves one call in real time with headroom; Whisper only runs while the caller speaks over the agent or just after their turn. No per-minute API cost. |

## 5. Results on the development set

Best real-time run on all 24 calls (`python -m ttbench.play --calls all`, run `cues-dev`, final rules of Nov 18):

```
Score 95.0 / 100   (stops when it should: 0.94, keeps talking when it should: 1.00)
21 right, 5 could be faster, 3 wrong (the 3 wrong are end-of-turn cases, not in the total)
```

Baseline B1: **41.0**. Our system: **95.0**: every real interruption stopped, no false stop. The frozen commit
(`d12ca5b`) gives the same 95.0 on a fresh run. A 30-minute soak (the 24 calls three times: clean, caller −12 dB, caller
+6 dB; 72 calls, one session, like the 64–80-scenario final) scored **94.3** with 0 false stops, so nothing degrades over
a long stream.

**What the words buy us.** The same server run with Whisper unavailable falls back to a loudness rule and scores **86.1**
(R 0.87, S 0.82: six false stops, on backchannels, background sound and a cough). Understanding the words adds about nine
points and removes every false stop on the dev calls. (We found this by accident: the server had been started without
its Python environment, which also showed us that the fallback never retries Whisper; that is on our fix list.)

**Dialect and noise (dev set, `python -m ttbench.summary`).** Najdi calls (12): R 0.92, S 1.00. Other dialects (12:
Hijazi, Eastern, Southern, Northern Saudi, Kuwaiti, Emirati, Qatari, Bahraini): R 0.94, S 1.00. Dialect gap −0.01, in
favour of the other dialects: our rules are lexical Gulf cues, not Najdi-specific, and Whisper large-v3-turbo handled
them well. Clean audio (13): F 0.98; noisy or phone (11): F 0.95, a noise gap of 0.03. The DV category (Kuwaiti, Emirati)
passes completely. Caveat: with 1–3 calls per category these gaps are not statistically solid; on our 58 team calls (more
voices, other scripts) R dropped to 0.53, so unseen speakers are the real risk, not dialect as such.

Because we tuned on these 24 calls, we also ran copies that we changed on purpose (only the caller channel, so labels
stay exact), to check that the system does not depend on how this recording happens to sound:

| Set (24 calls, real time) | Core | R | S |
|---|---|---|---|
| Clean dev calls | 95.0 | 0.94 | 1.00 |
| Caller 12 dB quieter | 93.5 | 0.90 | 1.00 |
| Caller 8 dB louder before the first event ("shouted hello"), stress test | 89.7 | 0.86 | 0.96 |
| Earlier, before the 400 ms lead-in: caller 6 dB quieter / louder | 87.5–88.6 / 89.9 | | |

The same code varies by about ±2 points between runs, because Whisper's first words on a burst vary.

**Unseen calls.** Our team also generated 58 short calls with other voices and scripts (28 scenarios, plus phone-line and
speakerphone-echo versions; labels checked with Whisper, text match 0.91). These calls have no "intent is clear" time, so
latency is counted from the caller's first syllable, which makes R look lower than it is. On them the current system
scores Core **71.0** (R 0.53, **S 1.00**: no false stop on any of the 30 KEEP calls, including echo, TV, café, horn and
door). This set showed us real misses that the dev calls never did: «استنى» heard as «ستنة» (now accepted), and the rising
«طيب؟» that text alone cannot separate from «طيب».

**A dialect-tuned recogniser, tested and not adopted.** `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` (same size
and speed as ours) heard first words better on the team calls (70.1 vs 65.7 before our rule fixes), but on the dev calls it
dropped to 88.8 with a false stop on TV speech and a missed «لحظة» (heard «لحفة»). We kept the base model and carried its
lessons into the rules («ستنى» variants, Gulf letters «گ/ڨ», stricter near-spelling cues): with them the base model
reaches 95.0 on dev and 71.0 on the team calls.

## 6. Where it fails (at least five cases)

| # | Call | What happened | Listen | Why it went wrong (your analysis) |
|---|---|---|---|---|
| 1 | OV-D03 | stopped 0.8 s after «يا أخي هذا ثالث مرة…» was clear (0.75 credit) | `python -m ttbench.render --run runs\pad-clean --scenario OV-D03` | At 0.6 s Whisper already has «يا أخي هذا», but «يا» does not count, so it is only 2 real words and our rule needs 3 (or a cue). Accepting 2 loud words would fix it, but on our logs it also stopped on TV speech, so we kept 3. |
| 2 | AM-D03 | kept talking over «خلاص خلاص، … فهمت» (preferred: stop; 0.6 credit) | `python -m ttbench.render --run runs\pad-clean --scenario AM-D03` | «خلاص» means both "OK, got it" (a backchannel) and "enough, stop". We list it as an acknowledgement, so the repeated, firm «خلاص خلاص» is read as agreement. Prosody (a firmer, louder repeat) would be needed to tell them apart. |
| 3 | GI-D03 | stopped 0.46 s after «لحظة» (0.92 credit) | `python -m ttbench.render --run runs\pad-clean --scenario GI-D03` | The first 0.4 s decode gave «لحفظ», a near miss of «لحظة»; the next decode, 100–300 ms later, was right. We do not loosen the fuzzy match further because it would also match unrelated words. |
| 4 | HT-D02 | took the turn while the caller was dictating an order number («لحظة أشوف» … «ثلاثة سبعة») | `python -m ttbench.render --run runs\pad-clean --scenario HT-D02` | Our end-of-turn rule sees «لحظة أشوف» and number strings as finished sentences. Waiting longer after «لحظة/أشوف» and after digits is drafted but untested (end-of-turn track only). |
| 5 | HT-D01 | took the turn at 19.4 s while the caller was still talking | `python -m ttbench.render --run runs\pad-clean --scenario HT-D01` | «يعني تقريبا ساعة سبعة» ends with a complete-looking phrase after a pause, so we took the turn after 0.7 s; the caller continued. Words alone cannot always tell a pause from an end. |
| 6 | BG-D01 | false stop on TV speech in the "loud start" stress copy («على المترجمات لكثير…») | `python -m ttbench.render --run runs\pad-loudstart+8 --scenario BG-D01` | The TV was louder than the caller early in the call, so it became the caller's "reference voice"; TV speech then looked as loud as the caller and passed. Text alone cannot tell a TV presenter from the caller. |
| 7 | live | (live microphone test, Nov 17) «اصبر، اصبر» did not stop the agent | demo recording | «اصبر» ("wait") is common Gulf Arabic but was not in our cue list (none of the 24 dev calls uses it). Added, together with 60 general Gulf floor-bid phrases checked against 2,719 logged decodes for false stops. |
| 8 | live | (live demo, Nov 19) the caller answered «يوم الثلوث» (Saudi for Tuesday) softly over the agent's question «الاثنين ولا الثلاثاء؟»; the agent finished its question instead of stopping | `demo_recordings/call-001356.wav` | Whisper heard nonsense («يوميسور أسا») and the burst was 8 dB quieter than the caller's own voice, so it was one or two unclear, quiet words: kept. A dialect word, said softly, at the end of a question. Letting 2 words stop the agent would fix it but caused false stops on TV in our replay. |

## 7. Strengths, weaknesses and trade-offs

**Strong:** backchannels, coughs, laughter and echo (no false stop in any dev or volume run); clear Gulf cues
(«لا لا», «لحظة», «اسمع», «بس», «اصبر») within ~0.2–0.5 s; quiet or loud callers and quiet lines (the gate is
relative to the caller); fully local, no network delay, no per-call cost.

**Weak:** speech from a TV or another person that is as loud as the caller; ambiguous words whose meaning is in the
tone («خلاص», «طيب»); interruptions without a cue word need 3 real words (~0.8 s); noisy phone lines make Whisper's
first words less reliable; end of turn during dictation or "let me check" pauses.

**Trade-offs we chose:** we prefer missing a little speed over a false stop (a false stop zeroes a KEEP event and
the category's restraint), so loosening rules is only accepted when a replay of all our logged decodes shows no new
false stop. We chose the 3-word rule and the −6 dB line partly by looking at the dev calls; the −12 dB line has a
10 dB margin, and every change was checked on the changed copies, not only on the dev calls.

**With another week:** prosody features (pitch, energy rise) for «خلاص/طيب»-type words; a speaker check (is this the
same voice as the caller?) against TV; `RESUME` to take back a wrong stop when the caller goes quiet within half a
second; more Gulf phrases collected from many speakers through the live demo.

## 8. External services, models and data

| Kind | Name and version | What for |
|---|---|---|
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (CTranslate2, float16) via `faster-whisper` | speech recognition in the scored system (local, GPU) |
| Library | faster-whisper, CTranslate2, NVIDIA cuBLAS / cuDNN 12 (pip wheels) | running Whisper on the GPU |
| Model | `Systran/faster-whisper-small` | early recognition probes only, not in the final system |
| API | ElevenLabs: `scribe_v3_live` (real-time) and `scribe_v1` (batch) speech-to-text | live demo only: the caller's words for the agent's reply (not in the scored system) |
| API | ElevenLabs text-to-speech: `eleven_v4_turbo`, `eleven_flash_v2_5`, built-in voices Sarah and Chris, and Saudi Voice Library voices (Abir, Hala, Maryam, Jumana) | live demo only: the agent's voice |
| API | Munsit (CNTXT AI): Faseeh `faseeh-v1-preview` text-to-speech (Najdi and Hijazi voices); `audio/transcribe` speech-to-text tested in comparison only | live demo option for the voice; the speech-to-text was a comparison (not used) |
| Tool | Cloudflare `cloudflared` quick tunnel | only to reach our laptop from the organisers' practice checkpoint |
| API | OpenRouter: `anthropic/claude-haiku-5.5` | live demo: writes the agent's replies; earlier, LLM STOP/KEEP judge experiment (dropped) |
| API | OpenRouter: `google/gemini-2.5-flash-lite` (writer), `google/gemini-2.5-flash` (transcription) | live demo options |
| API | OpenRouter: `openai/gpt-audio-mini` | live demo fallback voice (writes and speaks when ElevenLabs fails) |
| API | OpenRouter: `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-4.1-mini` | LLM STOP/KEEP judge experiment (dropped) |
| API | OpenRouter: `google/gemini-2.5-flash`, `google/gemini-3.7-flash`, `qwen/qwen3.8-omni-flash`, `mistralai/voxtral-small-24b-2507`, `openai/gpt-audio-mini` | comparison with cloud speech recognition (dropped) |
| Library | `edge-tts` (Microsoft neural ar-SA voices) | voice comparison for the demo (rejected) |
| Model | `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` (Apache 2.0), converted to CTranslate2 | tested as an alternative recogniser, not used in the final system |
| Data | Sarj development set (24 calls) and our own changed copies of it (caller volume ±6/12 dB, loud start, mic click) | development and stress tests |
| Data | Our team's generated calls (58 calls, ElevenLabs agent voices, recorded caller lines) | unseen test set only, never used to set numbers |
| Data | Our team's second batch (38 calls, ElevenLabs voices; `SARJ_TRAINING_DATASET/team_v4`) | built by the team for testing; not used for any number in this write-up. No private recordings or customer data. |

## 9. Team

Team 4 (for now :D): Usama, Rayan Bin Malik, Nada Alosaimi, Rana Alamri and Lena Alhaobani.
