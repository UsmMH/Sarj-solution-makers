# Write-up: Team 4 (for now :D)

## 1. The idea

**Approach, in a few words:** hybrid: acoustic gating + streaming speech recognition + Saudi/Gulf Arabic rules that read
both the caller's words and how loud they are compared with the caller's own voice.

An adaptive energy detector notices caller sound while the agent talks. The caller's audio is streamed live to
ElevenLabs speech-to-text (`scribe_v3_live`), and every partial transcript is judged the moment it arrives by Arabic
rules: a floor bid («لحظة», «لا لا», «اصبر», «ماني فاهم», a question, a real sentence) means STOP; a backchannel («إيه»,
«تمام»), an echo of the agent, side talk, TV or noise means keep talking. Loudness is judged *relative to this caller*,
so a soft speaker is not mistaken for background noise.

**Submitted system: `my_system_eleven.py`** (start: `python my_system_eleven.py --host 0.0.0.0`). It reuses the whole
pipeline of `my_system.py` (our local-Whisper version, kept as the alternative) and replaces only the recogniser. It is a
cloud system: it needs internet; without it a loudness-only rule takes over until the connection is back.

## 2. Why this approach

The baseline (B1, 41.0) stops on any loud sound. Our loudness-only rule reached 86.1 on dev but cannot tell «إيه» from
«لحظة»: the difference between "keep going" and "stop" is in the words. We built the rules on a local Whisper first
(dev 95.7), then measured recognisers on the same 121 caller events: ElevenLabs live had the most accurate text (0.90
match vs Whisper 0.86, Gemini Live 0.61) and never invents text on noise, while Whisper wrote YouTube subtitle phrases
(«ترجمة نانسي قنقر») on 14 of 121 short clips. Across 8 acoustic versions of the calls the two engines scored almost the
same (88.6 vs 89.8, within run-to-run noise), and ElevenLabs held up clearly better in room echo (90.2 vs 81.3). We submit
the engine whose text we trust most on unseen, noisy calls; its cost is speed (section 4).

## 3. How it works

```
caller audio, 20 ms frames (agent TEXT used for echo detection)
 ├─ adaptive noise floor (20th percentile of last 3 s + 10 dB) ── burst of caller sound while the agent talks?
 ├─ network thread (never blocks the 20 ms loop): caller audio streamed to scribe_v3_live in 100 ms messages
 │     every partial transcript of the open burst is judged at once; at the burst's end we "commit" and judge the
 │     final text too (single short words such as «لحظة» often arrive only then); commits wait for >= 0.32 s of audio
 ├─ decide(text, last 3 agent sentences, burst peak, caller reference voice):
 │     keep:  nothing heard · echo of the agent · side talk · burst >12 dB quieter than the caller (TV, echo)
 │     STOP:  cue word (لحظة، استنى، اسمع، بس، اصبر، «لا» at the start…) or one of 60 phrases
 │     STOP:  a question, or 3+ real words, unless more than 6 dB quieter than the caller's voice
 │     STOP?: 2 real words at the caller's own loudness = speculative STOP; RESUME if newer text does not confirm it
 └─ end of turn: caller quiet 300 ms → commit → text unfinished? wait up to 1.6 s, else 0.7 s → TAKE_TURN
Connection lost: loudness-only rule at once; reconnects every 2 s and switches back by itself.
```

The caller reference voice is the loudest frame of sustained caller speech so far in the call, outside the burst being
judged, reset per call. A live browser demo (`demo.py`) shows the system on a microphone; its other cloud parts (reply
writer, voice) are demo only.

## 4. Speed and cost

| | |
|---|---|
| Stop latency, dev (real time) | median 559 ms after the intent was evident, p90 858 ms (Whisper version: 426 / 507 ms) |
| Where the time goes | ElevenLabs' first partial text arrives ~0.7 s after the caller starts; final text ~0.25 s after a commit |
| Hardware | any laptop, no GPU; one websocket per call |
| Cloud calls and cost | ElevenLabs realtime speech-to-text: ~165 credits per 10 minutes of streamed audio (starter plan) |

## 5. Results

| Set (live, real time) | ElevenLabs (submitted) | Whisper version |
|---|---|---|
| Dev, 24 calls | 85.8 (R 0.76, S 1.00) | 95.7 (R 0.95, S 1.00) |
| Dev, 16 check calls, mean over 8 acoustic versions (below) | 88.6 | 89.8 |
| Organisers' checkpoint, 16 new calls (Nov 19) | not run | 77.0 (R 0.63, S 0.92) |
| Team data, locked test split, 11 calls, run once | not run | 74.7 (R 0.60, S 1.00) |

Baseline B1: 41.0. Live runs vary by about ±2 points. On all 24 dev calls the ElevenLabs version loses most of its gap
on a single call (OV-D01, section 6); 1 of 7 interruptions missed costs ~10 points on dev.

**Robustness on the 24 dev calls only.** Fixed before any result: the 8 `*-D01` calls may be inspected; the 16
`*-D02/D03` calls are only scored ("check fold"). Each call is also played in 7 label-safe copies where only the caller
channel changes, with textbook parameters (`augment.py`): −12 dB line, loud start, a click, telephone band 300–3400 Hz +
G.711, pink noise at 15 dB SNR, car-like brown noise at 5 dB SNR, room echo RT60 0.5 s (`eval_sets.py`).

| Check fold, 16 calls (Core) | dev | −12 dB | loud start | click | phone | pink | brown | echo | **mean** |
|---|---|---|---|---|---|---|---|---|---|
| ElevenLabs (submitted) | 88.9 | 89.8 | 89.3 | 89.1 | 88.4 | 85.3 | 87.6 | 90.2 | **88.6** |
| Whisper version | 92.9 | 91.1 | 89.6 | 92.9 | 89.0 | 88.4 | 93.5 | 81.3 | **89.8** |

Neither engine missed an interruption on the check fold in any copy. **These calls are not truly unseen:** our word
rules were first written (Nov 17) with all 24 dev transcripts available, so these are robustness numbers, not a forecast.
A clean rebuild whose rules were written without reading any transcript scored 77.5 on the same check fold, close to the
organisers' unseen checkpoint (77.0): we expect unseen calls in the high 70s, not near 89.

**Why dev and unseen differ.** Dev was read while the rules were written. The checkpoint caught 5 of 7 interruptions
(Wilson 95% CI [0.36, 0.92]): small samples swing a lot. Restraint is the strong side: 0 of 44 KEEP events failed on
dev + team data, and S = 1.00 on 6 of the 8 check-fold versions (phone 0.96, brown noise 0.94).

**Nov 20 changes, kept (both engines):** speculative STOP on two loud words with a RESUME guard (Whisper replays: 11
fires, all real interruptions, ~300 ms earlier, no RESUME needed); dialect cues («انطر», «معليش», «لو سمحت», negation
«مو/مب/مش» at the start, Hijazi and Kuwaiti question words): 2 real stops gained, 0 new false stops on 2,533 logged decodes.

## 5b. Tried and rejected

| Idea | Result |
|---|---|
| Hedged STOP at 200-300 ms on any loud sound, RESUME if it is an acknowledgement | S fell to 0.73-0.80: every loud KEEP burst became a half-credit hiccup |
| Gemini Live as the recogniser | first text 1.82 s median, nothing on 38 of 121 events; 0 of 56 stops by 1.0 s |
| Google Cloud Chirp streaming | needs a billing account we could not open from Saudi Arabia: not tested |
| Dialect-tuned Whisper (`oddadmix/...-arabic-dialectal-v2`) | +4 on team calls, but dev 88.8: a TV false stop, «لحظة» heard «لحفة» |
| Whisper `int8_float16`, Whisper small, first decode at 200 ms | no faster or more errors (validation 63.0 → 37.0 / −5.2 / −9.2) |
| Joining bursts split by a pause | validation swung +7 / −8 with the gap setting: fragile |
| Speaker-embedding veto, pitch/voicing classifier, LLM judge, confidence gates | enrolment too short; AUC 0.54; 2-4 false stops per model; false stops are confident decodes |

## 6. Where it fails

Dev set (24 calls), submitted system (`python -m ttbench.summary runs/eleven-dev --failures`), plus the phone copy:

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 1 | OV-D01 | Missed «إيه الأحد … زين، احجزه» (both started at once) | ElevenLabs heard it right, but a short pause split it into «إي، الأحد. زين.» and «احجزه.»; each piece is judged alone and never reaches 3 real words. Joining pieces across pauses was unstable when tested. |
| 2 | GI-D02 | Stopped 0.75 s after the interruption was clear | The first partial text arrives ~0.7 s after the caller starts: the main cost of this engine. |
| 3 | GI-D01 | Stopped 0.58 s after «لا لا … لحظة» was clear | Same: «لا، لا.» was the first partial, at ~0.6–0.9 s into the burst. |
| 4 | OV-D03 | Stopped 0.86 s after «يا أخي هذا ثالث مرة…» was clear | Partials arrive late; the 3-word rule fired on «يا أخي هذا سؤال-» at 1.2 s into the burst. |
| 5 | AM-D03 | «خلاص خلاص، … فهمت»: kept talking (preferred: stop; 0.6 credit) | «خلاص» means both "OK" and "enough"; the meaning is in the tone, which text loses. |
| 6 | HT-D02 | End-of-turn: took the turn three times while the caller dictated | Phrases such as «لحظة أشوف» and digit strings look finished to our end-of-turn rule (separate track). |
| 7 | BG-D03 | Phone copy: false stop on side speech (someone in the room, same dialect) | On a narrow phone band the other voice sounds as close as the caller: no text or loudness rule separates them. |

Unseen data and live tests (Whisper version; the rules are shared):

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 8 | live, Nov 17 | «اصبر، اصبر» did not stop the agent | «اصبر» was in no dev call; added with 60 general Gulf phrases, 0 new false stops on 2,719 logged decodes. |
| 9 | AM02 («طيب؟») | Never stopped | The rising tone is lost in text; «طيب» reads as an acknowledgement. |
| 10 | T4-BG01 (TV news) | False stop | Background speech at the caller's own loudness, transcribed fluently. |
| 11 | rebuild, noisy copy of OV-D01 | «لحظة» transcribed as «لحم» | In noise the cue word itself can be misheard; no spelling rule can safely accept «لحم». |

## 7. Strengths, weaknesses and trade-offs

**Strong:** backchannels, coughs, laughter and echo; room echo and phone audio; quiet or loud callers; no invented text
on noise; runs on any laptop. **Weak:** speed (first text ~0.7 s, so full credit is rare); sentences split by a pause;
speech from a TV or person as loud as the caller; words whose meaning is in the tone («خلاص», «طيب»); needs internet.
**Trade-off:** we prefer a late stop over a false stop (a false stop zeroes a KEEP event). The 3-word rule and the −6 dB
line were partly chosen while looking at dev; the −12 dB line has a 10 dB margin.

**With another week:** run Whisper's fast first look and ElevenLabs' accurate text together (each covers the other's
weakness); a keyword spotter for the first 300 ms; judging a sentence across a short pause, validated on unseen calls.

## 8. External services, models and data

**Submitted system:** ElevenLabs realtime speech-to-text `scribe_v3_live` (cloud API, in the scored path: its delay
counts), NumPy, websockets. No other model or service in the scored path.

| Kind | Name | Status |
|---|---|---|
| API | ElevenLabs `scribe_v3_live` (realtime speech-to-text) | **submitted system** (cloud) |
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (local GPU, faster-whisper) | Whisper alternative (`my_system.py`), not submitted |
| Model | Silero VAD (bundled in faster-whisper); `Systran/faster-whisper-small`; `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` | analysis / tested, not used |
| API | ElevenLabs `scribe_v1`, `eleven_v4_turbo`, `eleven_flash_v2_5` (text-to-speech) | demo only |
| API | Google AI Studio Gemini Live: `gemini-2.5-flash-native-audio-latest` | experiment only |
| API | Munsit (CNTXT AI): `faseeh-v1-preview` text-to-speech, `audio/transcribe` | demo / comparison only |
| API | OpenRouter: `anthropic/claude-haiku-5.5` (demo writer, LLM judge test), `google/gemini-2.5-flash-lite`, `google/gemini-2.5-flash`, `google/gemini-3.7-flash`, `openai/gpt-audio-mini`, `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-4.1-mini`, `qwen/qwen3.8-omni-flash`, `mistralai/voxtral-small-24b-2507` | demo and experiments only |
| Library / tool | `edge-tts` (rejected demo voice); Cloudflare `cloudflared` (practice checkpoint only) | not in the submitted system |
| Data | Sarj development set (24 calls) and our label-safe copies (volume, loud start, click, phone, noise, echo) | development and robustness tests |
| Data | Our team's generated calls (58 + 38; ElevenLabs agent voices, recorded caller lines) | held-out tests of the Whisper version; no private or customer data |

## 9. Team

Team 4 (for now :D): Usama, Rayan Bin Malik, Nada Alosaimi, Rana Alamri and Lena Alhaobani.
