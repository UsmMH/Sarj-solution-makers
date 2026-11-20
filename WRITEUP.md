# Write-up: Team 4 (for now :D)

## 1. The idea

**Approach, in a few words:** hybrid: acoustic gating + local speech recognition + Arabic rules that read both the
caller's words and how loud they are compared with the caller's own voice.

An adaptive energy detector notices caller sound while the agent talks. A local Whisper model (large-v3-turbo, laptop
GPU) re-reads that burst every ~300 ms, and Saudi/Gulf Arabic rules decide: a floor bid («لحظة», «لا لا», «اصبر»,
«ماني فاهم», a question, a real sentence) means STOP; a backchannel («إيه», «تمام»), an echo of the agent, side talk,
TV or noise means keep talking. We judge loudness *relative to this caller*, so a soft speaker is not mistaken for
background noise. For end of turn we wait longer when the caller's last words sound unfinished («و…», «يعني», a filler).

**Frozen scored code: commit `5783b18`** = the Nov 17 rules plus a robustness fix: Whisper is retried in the
background after failed decodes, with a loud startup warning. Decisions are identical to `d12ca5b` when Whisper works.

## 2. Why this approach

The baseline (B1, 41.0) stops on any loud sound. A loudness-only rule of ours reached 86.1 on dev but cannot tell
«إيه» from «لحظة»: the difference between "keep going" and "stop" is in the words. So we added recognition, then spent
most of our time on the places where it fails:

- **Whisper's "subtitle credit".** On short, abrupt clips Whisper often outputs «ترجمة نانسي قنقر» ("translated
  by …"), a phrase from its training subtitles. Giving it 400 ms of audio before the burst removed it from all 27 short
  stop clips we tested (was 6/27); clean dev went from 86.5–90.1 to 94.6.
- **A fixed loudness gate breaks on quiet callers** (caller 12 dB quieter: 74.4). We now compare each burst with the
  caller's own earlier voice (sustained speech only): 93.5 on the same quiet copy.
- **Live testing found what dev could not:** «اصبر» was missing from our cues; after a STOP on a short «لا» the agent
  stayed silent for 8 s; a headset noise gate chopped words.

## 3. How it works

```
caller audio, 20 ms frames (agent audio unused; agent TEXT used for echo detection)
 │
 ├─ adaptive noise floor (20th percentile of last 3 s + 10 dB) ── burst of caller sound? (only while the agent talks)
 │
 ├─ Whisper worker thread (never blocks the 20 ms loop): growing burst + 400 ms lead-in, first after 300 ms, then ~100 ms
 │
 ├─ decide(text, last 3 agent sentences, burst peak, caller reference voice):
 │     keep:  nothing heard · stock phrases · echo of the agent · side talk · burst >12 dB quieter than the caller
 │     STOP:  cue word (لحظة، استنى، اسمع، بس، اصبر، «لا» at the start…) or one of 60 phrases
 │     STOP:  a question, or 3+ real words, unless more than 6 dB quieter than the caller's voice
 │     STOP?: 2 real words at the caller's own loudness (−3 dB) = speculative STOP; RESUME if the next decode
 │            does not confirm it within 450 ms (Nov 20)
 │
 └─ end of turn (agent silent): caller quiet 300 ms → Whisper reads their last 4 s → unfinished? wait up to 1.6 s,
    else 0.7 s → TAKE_TURN. After our own STOP the caller has the floor, so TAKE_TURN follows their silence.
Fallback: if Whisper fails 3 times, or the median of the last 5 decodes is above 600 ms (busy or slow GPU), a
loudness-only rule decides, and Whisper is retried / probed in the background.
```

The caller reference voice is the loudest frame of sustained caller speech so far in the call, outside the burst being
judged, reset per call. A live browser demo (`demo.py`) runs the same system on a microphone; its cloud parts (live
transcription, reply writer, voice) are demo only and labelled on the page.

## 4. Speed and cost

| | |
|---|---|
| Median stop latency, dev set (real time) | 416 ms after the intent was evident (p90 754 ms) on the `d12ca5b` rules; the `5783b18` re-run gives 438 ms (p90 828 ms) |
| Where that time goes | 300 ms of burst before the first decode; Whisper decode ~280 ms; up to 100 ms waiting for the next decode slot |
| Hardware and size | RTX 3060 Laptop GPU (6 GB), ~2.4 GB used; mains power only (battery: decodes ~35 % slower); Whisper large-v3-turbo, ~809 M parameters |
| Cloud calls and cost | 0 calls, no per-minute cost; one mid-range GPU serves one call in real time with headroom |

## 5. Results

Frozen scored commit: 5783b18. Numbers measured on the `d12ca5b` rules (the volume copies below, the checkpoint) are
unchanged by the robustness fix.

| Set | Core | R | S |
|---|---|---|---|
| Dev, 24 calls (real-time re-run) | 94.6 | 0.93 | 1.00 |
| Team set, 58 calls | 70.4 | 0.52 | 1.00 |
| Team set, two «طيب؟» events relabelled "either" | 79.4 | 0.66 | 1.00 |
| Organisers' checkpoint, 16 new calls | 77.0 | 0.63 | 0.92 |
| 30-min soak on 5783b18 (72 calls, one session) | 93.3 | 0.90 | 1.00 |

Baseline B1: 41.0. The team set (58 calls, other voices and scripts, phone and echo versions) has no "intent is clear"
time, so latency counts from the caller's first syllable. Changed copies of dev, same rules: caller 12 dB quieter 93.5
(R 0.90, S 1.00); caller 8 dB louder before the first event 89.7 (S 0.96, BG-D01 below). Runs vary by about ±2 points.

**Why dev and unseen differ.** (1) Dev is synthetic and our cue rules were written after reading it, so 94.6 is an
optimistic number. (2) Small n: the checkpoint caught 5 of 7 interruptions, Wilson 95% CI [0.36, 0.92], compatible with
both the dev and the team results. (3) Label artifact: «طيب؟» heard as «طيب» is an acknowledgement by text but an
ambiguous case for the scorer; relabelling it lifts team R from 0.52 to 0.66 with no system change. (4) A latency
floor: the first Whisper decode lands 550-670 ms after onset (300 ms burst + ~280 ms decode), so full credit is rarely
possible.

**Where the credit goes** (33 genuine events, 10.88 credit lost): first decode a stock phrase or misheard 4.02; misses
3.00; cue word late in the utterance 2.35 (mostly legitimate); latency floor 1.51. A perfect first decode (oracle) would
add +2.8 to +6.9 points on seen data, so recognition quality is not a large hidden gap.

**Restraint.** 0 of 44 KEEP events failed on dev + team (Wilson 95% upper bound on the unseen false-stop rate about
8%); checkpoint Restraint 0.92, in line with the top teams.

Without Whisper the server falls back to a loudness rule: 86.1 on dev (six false stops). **Dialect Whisper, not adopted:** `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2`
scored 70.1 vs 65.7 on the team calls (before our rule fixes) but 88.8 on dev (a false stop on TV, «لحظة» heard
«لحفة»); we kept the base model and carried its lessons into the rules («ستنى» variants, Gulf letters «گ/ڨ»).

## 5b. What we tried after the checkpoint, and rejected (real scorer or offline replays)

| Idea | Result |
|---|---|
| Hedged early STOP at 200-300 ms, RESUME within 500 ms if the first decode looks like an acknowledgement | Loses points in all 9 settings: S falls to 0.73-0.80 because every loud KEEP burst becomes a half-credit hiccup; best case dScore -1.9 (CI -8.6 to +6.0), real-scorer reading -6.2 at best; if a RESUME lands over a real caller the event can score as a miss |
| Fallback: STOP when the first decode is empty or a stock phrase but the burst is loud and speech-like (Silero VAD) | 0 of 14 all-stock bursts were genuine; fires on 3 of 4 blind genuine events in-sample (all one scenario group, 0 of 4 when that group is held out) and on 1 of 17 KEEP bursts (a laugh); estimated gain +0.56, CI [-2.12, +3.72]; review flagged loud backchannels and TV as likely false stops. Implemented behind a flag, not shipped |
| Whisper confidence features | no_speech_prob is exactly 0.0 on everything; avg_logprob points the wrong way (stock phrases are the most confident decodes, AUC 0.07); echo score AUC 0.59 [0.35, 0.80] |
| Banning stock-phrase tokens | The decoder falls back to «شكرا»: 0 of 4 recovered |
| ElevenLabs `scribe_v3_live` as the recogniser | Measured offline: first text at 0.75 s (nothing before 0.7 s), same recall as Whisper, slower; used in the demo only |
| Earlier | Prosodic classifier (AUC 0.54), WeSpeaker speaker veto (too little clean speech to enrol), cloud STT (0.83 match in 2.4-3.9 s vs Whisper 0.75 in 0.5 s), cloud LLM judge (2-4 false stops per model), Whisper text prompt, beam-5 retry, 2-word rule, longer lead-in: none helped |

Lesson: with S = 1.00 on every seen set, "replay shows no new false stop" cannot validate a loosening, so we shipped the
system whose Restraint we could actually verify.

## 5c. Nov 20: generalisation check with a train / validation / locked-test split

The team dataset was split by assistant script and voice (no script, voice or caller clip in two splits): train 44
calls, validation 11, test 11. Dev was used as a sanity check only. Every change was first measured with a
deterministic virtual-clock replay of the real system (`sim.py`: the real Whisper; results land after their measured
decode time; it matched live runs within 0.4 points on four sets), then confirmed live. The team splits have no
"intent is clear" time, so their latency counts from the first syllable.

| Set (live, real time) | Before (frozen 5783b18) | After (Nov 20) |
|---|---|---|
| Dev, 24 calls (sanity only) | 94.6–95.0 | 95.7 (R 0.95, S 1.00) |
| Train, 44 calls | 62.4 (sim) | 64.2 (R 0.49, S 0.92) |
| Validation, 11 calls | 63.0 (sim) | 65.0 (R 0.45, S 1.00) |
| **Test, 11 calls, locked, run once** | 72.2 (earlier runs, re-scored) | **74.7 (R 0.60, S 1.00)** |
| Organisers' checkpoint (not re-run) | 77.0 (R 0.63, S 0.92) | — |

The gap between dev (~95) and everything unseen (63–77) is the honest picture: dev was read while the rules were
written. The Nov 20 changes are each smaller than live run-to-run variance (±2), so we claim "no worse anywhere, slightly
better", not a measured win. The test split is not pristine: earlier sessions had already scored these calls with the
frozen system (no rule was written from them), and the validation call IH05 is where the «ستنى» spellings came from.

Kept: (B) the speculative STOP on two loud words with a RESUME guard: in replays it fired 11 times, always on a real
interruption, about 300 ms earlier (OV-D03 «يا أخي هذا…», T4-GI03, T4-DV01, T4-GI12), and it never needed a RESUME;
(C) dialect cues («انطر/نطر», «معليش/معلش», «لو سمحت», «عن اذنك», negation at the start «مو/مب/موب/مش/ماهو» except
«مو مشكلة/مو بطال», Hijazi and Kuwaiti question words «امتى، فين، منو، شكثر»), Gulf ض/ظ merged, near-spellings
allowed across swapped letters (ت/د/ث…): no score change on our sets, 2 real live interruptions gained in a replay of
2,533 logged decodes, 0 new false stops; 60 unit tests written before any run (`test_dialect_rules.py`);
(F) a slow-decode watchdog: at three times the normal decode time the system would score 27.6 on dev with Whisper and
85.4 with the switch.
Rejected: `int8_float16` (no faster on this GPU, val 63.0 → 37.0); first decode at 200 ms; Whisper small alone or as a
first look (−160 ms per decode but more errors, val −5.2); joining bursts split by a pause (validation swung +7 / −8
depending on the gap); stripping stock phrases (in 591 such decodes only one hid a "cue", and it was false); confidence
gates (the false stops are confident decodes); an echo-aware floor (none of our calls has the agent in the caller
channel, so it could not be tested). A rule ablation found no rule group that helps on dev alone; the 3-word rule and
the −6 dB line help dev most and are the closest to dev-fitted (`analysis/REPORT_nov20.md`).

## 6. Where it fails

Dev set (24 calls), final system, from `python -m ttbench.summary runs/live-final-dev --failures` (no misses, no false
stops; these are late stops and end-of-turn errors), plus the one false stop of the loud-start stress copy:

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 1 | GI-D02 | Real interruption; stopped 0.5 s after it was clear (full credit needs 0.3 s) | Latency floor: no decode before the burst is 300 ms long, plus ~280 ms per decode, so the first possible STOP is ~0.55-0.65 s after the caller starts. A 200 ms first decode was tested and scored lower (short clips are misread). |
| 2 | GI-D03 | «لحظة»; stopped 0.4 s after it was clear | Same floor. In earlier runs the first decode was «لحفظ», a near miss of «لحظة»; loosening the fuzzy match further would also match unrelated words. |
| 3 | AM-D02 | Ambiguous word; stopped 0.5 s after it was clear | Same floor: the decision needs the words, and the words need one decode. |
| 4 | OV-D03 | Both started at once; stopped 0.5 s after «يا أخي هذا ثالث مرة…» was clear (0.8 s before Nov 20) | «يا أخي هذا» is 2 real words and the rule needs 3 or a cue. The Nov 20 speculative STOP (2 words as loud as the caller, taken back by RESUME if not confirmed) cut 0.3 s; the rest is the floor. |
| 5 | HT-D02 | End-of-turn: took the turn three times while the caller dictated or paused («لحظة أشوف» … «ثلاثة سبعة») | Such phrases and digit strings look like finished sentences to our end-of-turn words rule (separate track). |
| 6 | HT-D01 | End-of-turn: took the turn 0.6 s after the caller finished, at the edge of full credit | After a pause, «يعني تقريبا ساعة سبعة» looks complete; we wait 0.7 s on finished-looking text. Words alone cannot always tell a pause from an end. |
| 7 | BG-D01 | Loud-start stress copy (`runs/live-final-loudstart`): false stop on TV speech («على المترجمات لكثير…») | The TV was louder than the caller early on, so it became the caller's reference voice; TV speech then looked as loud as the caller. Text cannot tell a TV presenter from the caller. |

Team data and live tests (held-out calls, harsher labels: latency counted from the first syllable):

| # | Call | What happened | Why it went wrong |
|---|---|---|---|
| 8 | IH04_clean_phone («لا، مو صح») | Miss: stopped 1810 ms after the intent was evident (limit 1500) | The first three decodes were the stock phrase «ترجمة نانسي قنقر»; the cue «لا» was read at 1.8 s. |
| 9 | IH05_noisy-door_phone («استنى استنى») | Never stopped | «استنى» was heard as «سنة» / «سنة سنة» four times (not a cue), then the stock phrase. |
| 10 | AM02_clean, AM02_clean_phone («طيب؟») | Never stopped | The rising «طيب؟» was decoded «طيب», an acknowledgement by text; the scorer treats it as ambiguous, so it is as much a labelling issue as a miss. |
| 11 | live, Nov 17 | «اصبر، اصبر» did not stop the agent | «اصبر» was in no dev call; added with 60 general floor-bid phrases checked against 2,719 logged decodes. |
| 12 | T4-GI09 («تمام بس الموعد ما يناسبني») | Miss: the STOP came after the 1.5 s limit | «تمام» was heard «تنظر / تنان»; the caller's short pause split the sentence into two bursts, so «بس» was read only in the second one. Joining bursts across pauses was tested and was unstable. |
| 13 | T4-GI11 («وش يعني بدون رسوم؟») | Miss although the STOP was decided at ~0.8 s | The question is about the agent's last words and starts 0.7 s before the agent finishes: our STOP arrived after the agent had stopped talking, so it counts for nothing. |
| 14 | T4-GI06 («ليش كذا؟») | Never stopped | Whisper heard «يشكي ده»: the question word was lost, two words left, no cue. |
| 15 | T4-DV09 («إيش الكلام ده؟ أنا ما طلبت حاجة») | Stopped too late | «إيش» was heard «إيه» (an acknowledgement), so the question did not count; the stop came from the second sentence. |
| 16 | T4-BG01 (TV news «وفي نشرة اليوم أعلن…») | False stop | Background speech at the caller's own loudness, fluently decoded (confidence −0.16): no text, loudness or confidence rule separates it from the caller. |

## 7. Strengths, weaknesses and trade-offs

**Strong:** backchannels, coughs, laughter and echo; clear Gulf cues («لا لا», «لحظة», «اسمع», «بس», «اصبر») within
~0.2–0.5 s; quiet or loud callers; fully local. **Weak:** speech from a TV or person as loud as the caller; ambiguous
words whose meaning is in the tone («خلاص», «طيب»); interruptions without a cue word need 3 real words; noisy phone
lines; end of turn during dictation. **Trade-off:** we prefer missing speed over a false stop (it zeroes a KEEP event).
We chose the 3-word rule and the −6 dB line partly by looking at dev; the −12 dB line has a 10 dB margin.

**With another week:**
- a fast first-300-ms keyword spotter (CTC, about 1.2 GB GPU): latency and first-decode errors are the biggest remaining loss;
- the question-context gate (agent just asked «؟» -> 1-2 non-acknowledgement words may stop), which needs data with many answer-to-question events to validate;
- a scorer-aware use of RESUME, since the replay shows the half-credit hiccup is too expensive for loud KEEP bursts.

## 8. External services, models and data

**Scored system (local only):** faster-whisper large-v3-turbo (CTranslate2, float16) via `faster-whisper`, NumPy,
websockets, NVIDIA cuBLAS / cuDNN 12. No network call, no other model.

| Kind | Name | Status |
|---|---|---|
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | **scored system** (local, GPU) |
| Model | Silero VAD (`silero_vad_v6.onnx`, bundled in faster-whisper, CPU) | offline analysis and one unshipped experiment; NOT in the scored system |
| Model | `Systran/faster-whisper-small`; `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` (Apache 2.0, CTranslate2) | early probes / tested alternative, not in the scored system |
| API | ElevenLabs: `scribe_v3_live`, `scribe_v1` (speech-to-text); `eleven_v4_turbo`, `eleven_flash_v2_5` (text-to-speech; voices Sarah, Chris, Saudi Voice Library) | demo/experiment only, not in the scored system |
| API | Munsit (CNTXT AI): `faseeh-v1-preview` text-to-speech; `audio/transcribe` speech-to-text (comparison only) | demo/experiment only, not in the scored system |
| API | OpenRouter: `anthropic/claude-haiku-5.5` (demo reply writer; LLM judge experiment), `google/gemini-2.5-flash-lite`, `google/gemini-2.5-flash`, `google/gemini-3.7-flash`, `openai/gpt-audio-mini` (fallback voice), `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-4.1-mini`, `qwen/qwen3.8-omni-flash`, `mistralai/voxtral-small-24b-2507` | demo/experiment only, not in the scored system |
| Library | `edge-tts` (Microsoft ar-SA voices) | demo/experiment only (rejected), not in the scored system |
| Tool | Cloudflare `cloudflared` quick tunnel | only to reach our laptop from the organisers' practice checkpoint |
| Data | Sarj development set (24 calls) and our changed copies (volume ±6/12 dB, loud start, click) | development and stress tests |
| Data | Our team's generated calls: 58 (ElevenLabs agent voices, recorded caller lines) and a second batch of 38 | held-out tests, never used to set numbers. No private recordings or customer data. |

## 9. Team

Team 4 (for now :D): Usama, Rayan Bin Malik, Nada Alosaimi, Rana Alamri and Lena Alhaobani.
