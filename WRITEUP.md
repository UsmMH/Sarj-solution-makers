# Write-up: Team 4 (for now :D)

## 1. The idea

**Approach, in a few words:** hybrid: acoustic gating + local speech recognition + Saudi/Gulf Arabic rules that read
both the caller's words and how loud they are compared with the caller's own voice.

An adaptive energy detector notices caller sound while the agent talks. A local Whisper model (large-v3-turbo, laptop
GPU) re-reads that burst every ~300 ms, and Arabic rules decide: a floor bid («لحظة», «لا لا», «اصبر», «ماني فاهم», a
question, a real sentence) means STOP; a backchannel («إيه», «تمام»), an echo of the agent, side talk, TV or noise means
keep talking. Loudness is judged *relative to this caller*, so a soft speaker is not mistaken for background noise.

**Scored code** (`my_system.py`, `turn_rules.py`) last changed in commit `21472df` (Nov 20): speculative STOP + RESUME,
dialect cues, slow-decode watchdog. The handed-in commit adds only documentation and test tools on top of it.

## 2. Why this approach

The baseline (B1, 41.0) stops on any loud sound. Our loudness-only rule reached 86.1 on dev but cannot tell «إيه» from
«لحظة»: the difference between "keep going" and "stop" is in the words. So we added recognition, then spent most of our
time where it fails:

- **Whisper's "subtitle credit".** On short, abrupt clips Whisper outputs «ترجمة نانسي قنقر», a phrase from its
  training subtitles. 400 ms of audio before the burst removed it from all 27 short stop clips tested (was 6/27).
- **A fixed loudness gate breaks on quiet callers** (caller 12 dB quieter: 74.4). Comparing each burst with the
  caller's own sustained voice gives 93.5 on the same copy.
- **Live tests found what dev could not:** «اصبر» was missing from our cues; after a STOP on a short «لا» the agent
  stayed silent for 8 s; a headset noise gate chopped words.

## 3. How it works

```
caller audio, 20 ms frames (agent TEXT used for echo detection)
 ├─ adaptive noise floor (20th percentile of last 3 s + 10 dB) ── burst of caller sound while the agent talks?
 ├─ Whisper worker thread (never blocks the 20 ms loop): burst + 400 ms lead-in, first decode at 300 ms, then ~every 100 ms
 ├─ decide(text, last 3 agent sentences, burst peak, caller reference voice):
 │     keep:  nothing heard · stock phrases · echo of the agent · side talk · burst >12 dB quieter than the caller
 │     STOP:  cue word (لحظة، استنى، اسمع، بس، اصبر، «لا» at the start…) or one of 60 phrases
 │     STOP:  a question, or 3+ real words, unless more than 6 dB quieter than the caller's voice
 │     STOP?: 2 real words at the caller's own loudness = speculative STOP; RESUME if not confirmed within 450 ms
 └─ end of turn: caller quiet 300 ms → Whisper reads their last 4 s → unfinished? wait up to 1.6 s, else 0.7 s
Fallback: Whisper fails 3 times, or decodes get slow (median > 600 ms): loudness-only rule, Whisper retried in background.
```

The caller reference voice is the loudest frame of sustained caller speech so far in the call, outside the burst being
judged, reset per call. A live browser demo (`demo.py`) runs the same system on a microphone; its cloud parts are demo only.

## 4. Speed and cost

| | |
|---|---|
| Stop latency, dev (real time) | median 426 ms after the intent was evident, p90 507 ms |
| Where the time goes | 300 ms of burst before the first decode + ~280 ms decode + up to 100 ms for the next slot |
| Hardware | RTX 3060 Laptop GPU (6 GB), ~2.4 GB used, mains power (battery: decodes ~35 % slower); ~809 M parameters |
| Cloud calls and cost | none in the scored system |

## 5. Results

| Set (live, real time) | Core | R | S |
|---|---|---|---|
| Dev, 24 calls | 95.7 | 0.95 | 1.00 |
| Organisers' checkpoint, 16 new calls (earlier code, Nov 19) | 77.0 | 0.63 | 0.92 |
| Team data, locked test split, 11 calls, run once | 74.7 (was 72.2) | 0.60 | 1.00 |
| Team data, train 44 / validation 11 calls | 64.2 / 65.0 | 0.49 / 0.45 | 0.92 / 1.00 |

Baseline B1: 41.0. Team calls (other voices and scripts, phone and noise versions) have no "intent is clear" time, so
latency counts from the first syllable: harsher than the real scorer. Live runs vary by about ±2 points.

**Why dev and unseen differ.** Dev was read while the rules were written, so ~95 is optimistic. The checkpoint caught 5
of 7 interruptions: Wilson 95% CI [0.36, 0.92], so small samples swing a lot. A latency floor (first decode 550-670 ms
after onset) makes full credit rare. Restraint is the strong side: 0 of 44 KEEP events failed on dev + team data.

**Robustness on the 24 dev calls only, without fitting them.** A protocol fixed before any result: only the 8 `*-D01`
calls may be inspected; the 16 `*-D02/D03` calls are only scored ("check fold"). Each call is also played in 7 label-safe
copies where only the caller channel changes, with textbook parameters (`augment.py`): −12 dB line, loud start, a click,
telephone band 300–3400 Hz + G.711, pink noise at 15 dB SNR, car-like brown noise at 5 dB SNR, room echo RT60 0.5 s.

| Check fold, 16 calls (Core) | dev | −12 dB | loud start | click | phone | pink | brown | echo | **mean** |
|---|---|---|---|---|---|---|---|---|---|
| Whisper (submitted) | 92.9 | 91.1 | 89.6 | 92.9 | 89.0 | 88.4 | 93.5 | 81.3 | **89.8** |
| ElevenLabs live (experiment) | 88.9 | 89.8 | 89.3 | 89.1 | 88.4 | 85.3 | 87.6 | 90.2 | **88.6** |

No interruption was missed on the check fold in any copy; one false stop each on the phone copy. Room echo is the
submitted system's weakest condition (stops slow to p90 1.3 s); on the inspect fold, phone and noise copies each cost
one interruption. The ElevenLabs version (`my_system_eleven.py`, same rules) is not submitted: cloud-dependent, slower on
clean audio, and 1.2 points lower on average (within the ±2 spread).

**These calls are not truly unseen:** the word rules were first written (Nov 17) with all 24 dev transcripts available,
so the table measures robustness, not unseen-call performance. A clean ElevenLabs rebuild whose rules were written
without reading any transcript scored 77.5 on the same check fold, close to the organisers' unseen checkpoint (77.0):
we expect unseen calls in the high 70s.

**Nov 20 changes, kept:** (B) speculative STOP on two loud words with a RESUME guard: fired 11 times in replays, always
on a real interruption, ~300 ms earlier, never needed a RESUME; (C) dialect cues («انطر», «معليش», «لو سمحت», negation
«مو/مب/مش» at the start, Hijazi and Kuwaiti question words), checked against 2,533 logged decodes: 2 real stops gained,
0 new false stops; (F) slow-decode watchdog: at 3× decode time dev would fall to 27.6, with the switch 85.4.

## 5b. Tried and rejected

| Idea | Result |
|---|---|
| Hedged STOP at 200-300 ms on any loud sound, RESUME if it is an acknowledgement | S fell to 0.73-0.80: every loud KEEP burst became a half-credit hiccup |
| "Default stop", as the top checkpoint teams' R suggests: 1 s of speech at the caller's own loudness with one non-backchannel word = STOP (`my_system_ds.py`, simulated) | Check-fold mean 90.1 vs 90.1: catches the overlap in noise (OV-D01 0 → 0.84), echo +4.6, pink noise +3.5, but a new false stop (OV-D02) on 4 of 8 copies; brown noise −4.6. Rejected by our pre-set rule |
| ElevenLabs `scribe_v3_live` as the recogniser | Best text accuracy (0.90 vs Whisper 0.86) but first text ~0.75 s; full system 88.6 vs 89.8 (above) |
| Gemini Live as the recogniser | first text 1.82 s median, nothing on 38 of 121 events; 0 of 56 stops by 1.0 s (Whisper 25) |
| Dialect-tuned Whisper (`oddadmix/...-arabic-dialectal-v2`) | +4 on team calls, but dev 88.8: a TV false stop, «لحظة» heard «لحفة» |
| `int8_float16`, Whisper small, first decode at 200 ms | no faster or more errors (validation 63.0 → 37.0 / −5.2 / −9.2) |
| Joining bursts split by a pause | validation swung +7 / −8 with the gap setting: fragile |
| Speaker-embedding veto, pitch/voicing classifier, LLM judge, confidence gates, banning stock tokens | enrolment too short; AUC 0.54; 2-4 false stops per model; false stops are confident decodes; decoder falls back to «شكرا» |

Lesson: with S = 1.00 on every seen set, "no new false stop in a replay" cannot validate a loosening, so we shipped the
system whose Restraint we could verify.

## 6. Where it fails

Dev set (24 calls), final system (`python -m ttbench.summary runs/live-final-dev --failures`: no misses, no false
stops), plus the one false stop of the loud-start copy:

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 1 | GI-D02 | Stopped 0.5 s after the interruption was clear (full credit needs 0.3 s) | Latency floor: no decode before 300 ms of burst, plus ~280 ms per decode. A 200 ms first decode scored lower (short clips are misread). |
| 2 | GI-D03 | «لحظة»; stopped 0.4 s after it was clear | Same floor; in earlier runs the first decode was «لحفظ», a near miss we do not loosen the match for. |
| 3 | AM-D02 | Ambiguous word; stopped 0.5 s after it was clear | Same floor: the decision needs the words, and the words need one decode. |
| 4 | OV-D03 | Both started at once; stopped 0.5 s after «يا أخي هذا ثالث مرة…» was clear (0.8 s before Nov 20) | «يا أخي هذا» is 2 real words; the speculative STOP cut 0.3 s, the rest is the floor. |
| 5 | HT-D02 | End-of-turn: took the turn three times while the caller dictated («لحظة أشوف» … «ثلاثة سبعة») | Such phrases and digit strings look finished to our end-of-turn rule (separate track). |
| 6 | HT-D01 | End-of-turn: took the turn 0.6 s after the caller finished, at the edge of full credit | «يعني تقريبا ساعة سبعة» looks complete after a pause; words alone cannot always tell a pause from an end. |
| 7 | BG-D01 | Loud-start copy: false stop on TV speech («على المترجمات لكثير…») | The TV was louder than the caller early on and became the reference voice. Text cannot tell a TV presenter from the caller. |

Unseen data (team calls, live tests):

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 8 | IH04_clean_phone («لا، مو صح») | Stopped 1.8 s after the intent (limit 1.5 s) | Three first decodes were the stock phrase «ترجمة نانسي قنقر». |
| 9 | IH05_noisy-door_phone («استنى استنى») | Never stopped | Heard «سنة سنة» four times, then the stock phrase. |
| 10 | AM02 («طيب؟») | Never stopped | The rising tone is lost in text; «طيب» reads as an acknowledgement. |
| 11 | live, Nov 17 | «اصبر، اصبر» did not stop the agent | «اصبر» was in no dev call; added with 60 general Gulf phrases, 0 new false stops on 2,719 logged decodes. |
| 12 | T4-BG01 (TV news) | False stop | Background speech at the caller's own loudness, decoded fluently: no text or loudness rule separates it. |

## 7. Strengths, weaknesses and trade-offs

**Strong:** backchannels, coughs, laughter and echo; clear Gulf cues within ~0.2–0.5 s; quiet or loud callers; fully
local. **Weak:** speech from a TV or person as loud as the caller; words whose meaning is in the tone («خلاص», «طيب»);
interruptions without a cue need 3 real words; noisy phone lines; room echo; end of turn during dictation.
**Trade-off:** we prefer a late stop over a false stop (a false stop zeroes a KEEP event). The 3-word rule and the −6 dB
line were partly chosen while looking at dev; the −12 dB line has a 10 dB margin.

**With another week:** a fast keyword spotter for the first 300 ms (latency and first-decode errors are the biggest
remaining loss); combining ElevenLabs' accuracy and echo robustness with Whisper's speed; a question-context gate (the
agent just asked «؟»), which needs more answer-to-question data to validate.

## 8. External services, models and data

**Scored system (local only):** faster-whisper large-v3-turbo (CTranslate2, float16), NumPy, websockets, NVIDIA cuBLAS /
cuDNN 12. No network call, no other model.

| Kind | Name | Status |
|---|---|---|
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | **scored system** (local, GPU) |
| Model | Silero VAD (bundled in faster-whisper, CPU) | offline analysis only |
| Model | `Systran/faster-whisper-small`; `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` (Apache 2.0) | tested, not used |
| API | ElevenLabs: `scribe_v3_live`, `scribe_v1` (speech-to-text); `eleven_v4_turbo`, `eleven_flash_v2_5` (text-to-speech) | demo and experiments only |
| API | Google AI Studio Gemini Live: `gemini-2.5-flash-native-audio-latest` | experiment only |
| API | Munsit (CNTXT AI): `faseeh-v1-preview` text-to-speech, `audio/transcribe` | demo / comparison only |
| API | OpenRouter: `anthropic/claude-haiku-5.5` (demo writer, LLM judge test), `google/gemini-2.5-flash-lite`, `google/gemini-2.5-flash`, `google/gemini-3.7-flash`, `openai/gpt-audio-mini`, `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-4.1-mini`, `qwen/qwen3.8-omni-flash`, `mistralai/voxtral-small-24b-2507` | demo and experiments only |
| Library / tool | `edge-tts` (rejected demo voice); Cloudflare `cloudflared` (practice checkpoint only) | not in the scored system |
| Data | Sarj development set (24 calls) and our label-safe copies (volume, loud start, click, phone, noise, echo) | development and robustness tests |
| Data | Our team's generated calls (58 + 38; ElevenLabs agent voices, recorded caller lines) | held-out tests, never used to set numbers; no private or customer data |

## 9. Team

Team 4 (for now :D): Usama, Rayan Bin Malik, Nada Alosaimi, Rana Alamri and Lena Alhaobani.
