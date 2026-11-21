# Write-up: Team 4

## 1. The idea

**Approach, in a few words:** hybrid: acoustic gating + local speech recognition + Saudi/Gulf Arabic rules that read
both the caller's words and how loud they are compared with the caller's own voice.

An adaptive energy detector notices caller sound while the agent talks. A local Whisper model (large-v3-turbo, laptop
GPU) re-reads that burst every ~300 ms, and Arabic rules decide: a floor bid («لحظة», «لا لا», «اصبر», «ماني فاهم», a
question, a real sentence) means STOP; a backchannel («إيه», «تمام»), an echo of the agent, side talk, TV or noise means
keep talking. Loudness is judged *relative to this caller*, so a soft speaker is not mistaken for background noise.
A second, cloud recogniser (ElevenLabs live) listens in parallel and may only *add* a STOP, only on a stop cue, where
Whisper heard nothing useful ("rescue").

**Scored code:** `my_system_hybrid_cue.py` (= `my_system.py` + cue-only rescue) and `turn_rules.py`. Every change was
measured before it was adopted (section 5). Run: `python my_system_hybrid_cue.py`.

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
 │     keep:  nothing heard · stock phrases · echo of the agent · side talk («يمه», «يا محمد…» unless the agent said
 │            that name) · burst >12 dB quieter than the caller
 │     STOP:  cue word (لحظة، استنى، اسمع، بس، اصبر، «لا» at the start…) or one of 60 phrases
 │     STOP:  a question (one word only if Whisper marks «؟» and it is as loud as the caller), or 3+ real words,
 │            unless more than 6 dB quieter than the caller's voice
 │     STOP?: 2 real words at the caller's own loudness = speculative STOP; RESUME if not confirmed within 450 ms
 │     STOP:  "default stop": 1 s of speech as loud as the caller's own voice with any non-backchannel word
 ├─ rescue: ElevenLabs live text of the same burst → same decide(); may STOP only on a stop cue/phrase, and only if
 │          Whisper's own reading was empty, a stock phrase or "too little to tell" (never overrules Whisper's vetoes);
 │          a bare «لا» waits for the burst's final text and never stops before a polite ending («لا لا إن شاء الله»)
 └─ end of turn: caller quiet 300 ms → Whisper reads their last 4 s → unfinished? wait up to 1.6 s, else 0.7 s
Fallback: Whisper fails 3 times, or decodes get slow (median > 600 ms): loudness-only rule, Whisper retried in background.
No internet / ElevenLabs down: no rescue, the Whisper system exactly (it never waits for the cloud).
```

The caller reference voice is the loudest frame of sustained caller speech so far in the call, outside the burst being
judged, reset per call. A live browser demo (`demo.py`) runs the same system on a microphone; its cloud parts are demo only.

## 4. Speed and cost

| | |
|---|---|
| Stop latency, dev (real time) | median 325 ms after the intent was evident, p90 514 ms (Whisper alone 476 / 525) |
| Where the time goes | 300 ms of burst before the first decode + ~280 ms decode + up to 100 ms for the next slot |
| Hardware | RTX 3060 Laptop GPU (6 GB), ~2.4 GB used, mains power (battery: decodes ~35 % slower); ~809 M parameters |
| Cloud | ElevenLabs `scribe_v3_live`, caller channel only; its text arrives ~0.7 s after speech, so it only adds missed stops |

## 5. Results

| Set (live, real time) | Core | R | S |
|---|---|---|---|
| Dev, 24 calls | 98.1 (Whisper alone 97.3) | 0.96 | 1.00 |
| Organisers' checkpoint, 16 new calls (earlier code, Nov 19) | 77.0 | 0.63 | 0.92 |
| Team data, locked test split, 11 calls, run once (Nov 20 code) | 74.7 (was 72.2) | 0.60 | 1.00 |
| Team data, train 44 / validation 11 calls | 65.0 / 65.4 | 0.51 / 0.46 | 0.90 / 1.00 |

Baseline B1: 41.0. Team calls (other voices and scripts, phone and noise versions) have no "intent is clear" time, so
latency counts from the first syllable: harsher than the real scorer. Live runs vary by about ±1–2 points. The
checkpoint caught 5 of 7 interruptions (Wilson 95% CI [0.36, 0.92]): small samples swing a lot.

**Robustness without fitting the dev calls.** A protocol fixed before any result: only the 8 `*-D01` calls may be
inspected; the 16 `*-D02/D03` calls are only scored ("check fold"). Each call is also played in 7 label-safe copies
where only the caller channel changes (`augment.py`): −12 dB line, loud start, a click, telephone band 300–3400 Hz +
G.711, pink noise at 15 dB SNR, car-like brown noise at 5 dB SNR, room echo RT60 0.5 s.

| Check fold, 16 calls (Core) | dev | −12 dB | loud start | click | phone | pink | brown | echo | **mean** |
|---|---|---|---|---|---|---|---|---|---|
| Whisper, Nov 20 rules | 92.9 | 91.1 | 89.6 | 92.9 | 89.0 | 88.4 | 93.5 | 81.3 | **89.8** |
| + default stop | 96.2 | 94.6 | 89.7 | 95.5 | 93.2 | 93.7 | 93.6 | 85.2 | **92.7** |
| + ElevenLabs rescue (2 runs) | 96.9 | 95.5 | 92.8 | 97.0 | 94.2 | 93.8 | 94.9 | 90.3 | **94.4** |
| **+ rescue on cues only (submitted)** | 97.5 | 96.3 | 93.0 | 97.4 | 93.8 | 97.0 | 95.0 | 90.5 | **95.1** |

No interruption was missed on the check fold in any copy; one false stop each on the phone copy, in every row. The word
rules were first written with the dev transcripts available, so this measures robustness, not unseen-call performance;
for unseen calls we expect the high 70s to mid 80s.

**How each change was adopted.** For each one we wrote the variant and its acceptance rule down *before* running it
(`analysis/PREREG_*.md`), and kept it only if the check-fold mean did not fall, no copy fell by more than 2 points, and
there were no new false stops (also on team data and a polite-«لا» set):
- **Speculative STOP + RESUME, dialect cues, slow-decode watchdog (Nov 20).** The speculative STOP fired 11 times in
  replays, always on a real interruption, ~300 ms earlier; the dialect cues («انطر», «معليش», «مو/مب/مش» at the start)
  gained 2 stops and 0 false stops on 2,533 logged decodes; the watchdog keeps 85.4 on dev when decodes are 3× slower.
- **Default stop (Nov 21).** The top checkpoint teams had our Restraint but far higher Responsiveness (R 0.87–0.97 vs
  0.63). Stopping on 1 s of speech at the caller's own loudness with any non-backchannel word: 89.8 → 92.7, no new
  false stop. It catches the ambiguous «خلاص خلاص… فهمت» and the overlap «إيه الأحد … زين، احجزه» in noise.
- **ElevenLabs rescue (Nov 21).** Whisper is fast but sometimes blind (stock phrases, echo); ElevenLabs is slower but
  never invents subtitle text. 92.7 → 94.0 / 94.7 in two runs. But in two runs it also read TV news as a sentence
  («في الشأن المحلي، أعلنت»), so the **submitted version lets it stop only on a cue or phrase**: equal score (95.1, within
  run noise), and all 14 of its stops were real cues («لحظة», «بس», «عفوا», «استنى», «اسمع»).

## 5b. Tried and rejected

| Idea | Result |
|---|---|
| ElevenLabs `scribe_v3_live` *instead of* Whisper | best text accuracy (0.90 vs 0.86) but first text ~0.75 s: 88.6 vs 89.8; much better in echo (90.2 vs 81.3), which led to the rescue |
| Hedged STOP at 200-300 ms on any loud sound, RESUME if it is an acknowledgement | S fell to 0.73-0.80: every loud KEEP burst became a half-credit hiccup |
| Dialect-tuned Whisper (`oddadmix/...-arabic-dialectal-v2`) | +4 on team calls, but dev 88.8: a TV false stop, «لحظة» heard «لحفة» |
| Speaker-embedding veto, pitch/voicing classifier, LLM judge, confidence gates | enrolment too short; AUC 0.54; 2-4 false stops per model; false stops are confident decodes |

## 6. Where it fails

Dev set (24 calls), final system: no misses, no false stops; the losses are latency and end of turn:

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 1 | GI-D02 | Stopped 0.5 s after the interruption was clear (full credit needs 0.3 s) | Latency floor: no decode before 300 ms of burst, plus ~280 ms per decode; a 200 ms first decode misreads short clips. |
| 2 | GI-D03 | «لحظة»; stopped 0.4 s after it was clear | Same floor; in earlier runs the first decode was «لحفظ», a near miss we do not loosen the match for. |
| 3 | OV-D03 | Both started at once; stopped 0.5 s after «يا أخي هذا ثالث مرة…» was clear | «يا أخي هذا» is 2 real words; the speculative STOP cut 0.3 s, the rest is the floor. |
| 4 | HT-D02 | End of turn: took the turn three times while the caller dictated («لحظة أشوف» … «ثلاثة سبعة») | Such phrases and digit strings look finished to our end-of-turn rule. |
| 5 | BG-D01 | Loud-start copy: false stop on TV speech («على المترجمات لكثير…») | The TV was louder than the caller early on and became the reference voice. |

Unseen data (team calls, live tests):

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 6 | IH04 phone («لا، مو صح») | Stopped 1.8 s after the intent (limit 1.5 s) | Three first decodes were the stock phrase «ترجمة نانسي قنقر». |
| 7 | IH05 noisy door, phone («استنى استنى») | Never stopped | Heard «سنة سنة» four times, then the stock phrase. |
| 8 | AM02 («طيب؟») | Never stopped | The rising tone is lost in text; «طيب» reads as an acknowledgement. |
| 9 | live, Nov 17 | «اصبر، اصبر» did not stop the agent | «اصبر» was in no dev call; added with 60 general Gulf phrases, 0 new false stops on 2,719 logged decodes. |
| 10 | T4-BG01 (TV news) | False stop | Background speech at the caller's own loudness, decoded fluently: no text or loudness rule separates it. |
| 11 | Polite «لا لا إن شاء الله» ("not at all") | False stop on all 3 polite calls (S 0.67) | Whisper reads «لا لا» first, about 0.7 s in; the polite ending arrives about 1.4 s in, too late to take the stop back. Waiting for it made real stops late (team data 71.0 → 66.4), so we keep the stop. |

## 7. Strengths, weaknesses and trade-offs

**Strong:** backchannels, coughs, laughter and echo; clear Gulf cues within ~0.2–0.5 s; quiet or loud callers; works
without internet (rescue off). **Weak:** a TV or person as loud as the caller (even one saying «لحظة»); words whose meaning is in the tone
(«خلاص», «طيب»); interruptions without a cue need 3 real words; noisy phone lines; end of turn during dictation.
**Trade-off:** we prefer a late stop over a false stop (a false stop zeroes a KEEP event). The 3-word rule and the −6 dB
line were partly chosen while looking at dev; the −12 dB line has a 10 dB margin.

**What we can add next:** a fast keyword spotter for the first 300 ms (latency and first-decode errors are the biggest
remaining loss); a question-context gate (the agent just asked «؟»).

## 8. External services, models and data

**Scored system:** the two rows marked below, plus NumPy, websockets and NVIDIA cuBLAS / cuDNN 12.

| Kind | Name | Status |
|---|---|---|
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | **scored system** (local, GPU) |
| API | ElevenLabs `scribe_v3_live` (speech-to-text, streamed) | **scored system** (rescue STOPs) and demo |
| Data | Our team's generated calls (58 + 38; ElevenLabs agent voices, recorded caller lines) | held-out tests only; no private or customer data |

Tried, never in the scored system (demo and experiments only): Whisper small, a dialect Whisper (oddadmix), Silero
VAD; ElevenLabs scribe_v1 and TTS (v4 Turbo, Flash v2.5); Munsit faseeh-v1; Gemini Live; via OpenRouter: Claude Haiku
5.5, Gemini 2.5/3.7 Flash, GPT-audio-mini, GPT-4.1 mini, Llama 3.3 70B, Qwen 3.8 omni, Voxtral small; edge-tts.

## 9. Team

Team 4: Usama, Rayan Bin Malik, Nada Alosaimi, Rana Alamri and Lena Alhaobani.
