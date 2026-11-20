# CLAUDE.md — Team 4, Sarj Arabic Turn-Taking Hackathon

Read this first. Hand-over from the chat sessions that built the system (Nov 15-19, 2026). Last update: afternoon of
Nov 19 (Day 5). Code freeze and final submission: **Saturday Nov 21, 12:00** (moved from Nov 19).

## Who and what
- User: Usama, final-year Information Systems student at KSU, working with a small team (some non-technical). He wants to go **step by step**, and wants **full git commands** (`git add ...` + `git commit -m "..."`), never just a message. He runs on Windows / PowerShell, RTX 3060 Ti laptop.
- Event: Sarj AI Human-Like Turn Taking Hackathon, Arabic phone-call agent. 5 teams. Our system decides, every 20 ms, whether the agent should STOP, RESUME or TAKE_TURN.
- Final ranking = 80 % Core score on a **hidden Arabic stream (Day 6, Nov 20)** + 20 % judges (originality 10, practical applicability 5, clarity of demo 5). Day 6 also has an unscored live-microphone demo the judges watch.
- Deadlines: **code freeze AND final submission Saturday Nov 21, 12:00** (moved from Nov 19/20; `python -m ttbench.ready` prints commit hash + service list). The connection check is most likely not happening (user, Nov 18): don't plan around it, but still test the system on the venue network before the final.
- The Day 3 checkpoint did not happen as a test; mentors gave advice instead: no new data recording (augment the 24 calls), cloud APIs are allowed (they count in latency, must be declared).
- Team name constant in `my_system.py` must stay: `TEAM = "Team 4 (for now :D)"`. Also keep `PEAK_DB = -20.0`.
- Cost-conscious with OpenRouter: good models, but watch the price (key limit $1; ~$0.05 used by Nov 18).

## Repo layout (this folder = the git repo, branch `main`)
- `my_system.py` — the scored system: adaptive-floor energy gate + local faster-whisper worker thread + end-of-turn probe. Never blocks the 20 ms loop.
- `turn_rules.py` — Arabic text rules: `decide(text, agent_text, peak_db, ref_db)` -> (STOP?, reason); `eot_incomplete(text)`; `worth_second_opinion()` (for the shelved LLM judge).
- `augment.py` — makes `data/aug/*` copies of the dev calls (only the caller channel changes, labels stay exact): volume sets `vol-12`, `vol-6`, `vol+6`; stress sets `loudstart+8`, `click` (`python augment.py --gains --stress`).
- `demo.py` + `demo/index.html` — live browser demo (see "Demo"). `demo_recordings/`, `demo_log.txt`, `demo/voice_samples/` are git-ignored.
- Shelved experiments (kept, not used by the system): `llm_judge.py` + `llm_replay.py` (OpenRouter LLM as STOP/KEEP judge), `asr_compare.py` (local Whisper vs cloud audio models).
- `eval_rules.py`, `asr_probe.py`, `diag.py`, `events_diag.py` — older offline helpers.
- `eot_v2_candidate/` — **untested** end-of-turn variant (see below), not part of the system.
- Starter-kit files (do not edit): CHALLENGE/DATA/INTERFACE/README/SCORING/SCRIPTS/SUBMISSION/TROUBLESHOOTING/WRITEUP .md, `ttbench/`, `scenarios/`, `schema/`, `tests/`, `data/dev/`, `examples/`.
- Secrets: `OPENROUTER_API_KEY` lives in `.env` (git-ignored; `llm_judge._read_key()` reads it). Never commit it.

## How to run (always real time — `--fast` is invalid with ASR; always PLUGGED IN — on battery Whisper is ~35 % slower and scores drop)
```
cd C:\Users\ososh\Work\Sarj_makers\turn-taking
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass      # once per window
.\.venv\Scripts\Activate.ps1
# window 1
python my_system.py                  # loads Whisper, then listens on ws://127.0.0.1:8765 (check the "[asr] ... ready" line)
# window 2
python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --name <name>                    # 24 calls, ~10 min
python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --data data/aug/vol-12 --name <name>   # an augmented set
```
- Only one Whisper process on the GPU at a time: stop `demo.py` before a benchmark run (two models share the 6 GB GPU and slow each other down).
- Env switches: `$env:ASR="off"` (energy-only fallback; **remember `Remove-Item Env:ASR`**), `$env:EOT="off"`, `$env:ASR_WARM="off"`, `$env:ASR_MODEL="small"`.
- `asr_log.txt` appends across runs; note its line count before a run to slice it. `[asr] burst ...` lines now print `vs caller ±N dB` (burst peak vs the caller's own voice). Scenario order in the log = GI-D01..DV-D03.
- Packages beyond requirements.txt: `faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`; `edge-tts` was installed for a voice test (unused).
- Reports: `runs/<name>/report.md`; `runs/history.csv`.

## Scoring (from SCORING.md)
- STOP: full credit if stopped <=300 ms after "evident", linear to 0.4 at 1500 ms, later = miss. KEEP (backchannel/noise/hesitation/echo): stopping = 0; a stop taken back by RESUME within 500 ms = 0.5. EITHER: preferred 1.0, other 0.6.
- R = mean STOP credit per category; S = KEEP credit x "no stray stop" per category; F = harmonic(R,S); Core = 100*(0.9F + 0.1A).
- End-of-turn score = its own leaderboard, NOT in the ranking. Don't over-invest.

## Pipeline (current)
1. Adaptive noise floor (20th pct of last 3 s + 10 dB, min -42 dB) flags a caller burst while the agent talks.
2. Worker thread transcribes the growing burst with Whisper large-v3-turbo (float16, CUDA, ar, beam 1, ~270-300 ms/decode, GPU kept warm).
3. **Caller reference voice** (`_ref_db`): the loudest frame of *sustained* caller sound (>= half of a 200 ms window loud), outside the burst being judged, reset per call. A click/bump on the mic does not set it.
4. `decide()` in order: nothing heard / Whisper stock phrase / echo of last 3 agent sentences / side talk -> keep; **burst > 12 dB quieter than the caller's voice -> keep even with a cue** (TV, speakerphone echo: Whisper heard «بس» in echo); stop cues (لحظة, استنى, اسمع, بس, لا at start, **اصبر/اصبري/اصبروا** ...) -> STOP; question or >=3 content words -> STOP unless > 6 dB quieter than the caller. Before any reference exists: the old absolute -20 dB gate.
5. Fallback: if ASR is missing or fails 3 times, energy rule with peak gate (-20 dB).
6. End of turn: caller quiet 300 ms -> decode last 4 s -> if the text ends mid-thought wait up to 1.6 s, else 0.7 s. **After our own STOP the caller has the floor**: TAKE_TURN follows their silence even if they went quiet before the STOP landed (a short «لا» used to leave 8 s of dead air).

## Results (Core on 24 calls; same code varies ~±2 run to run)
| set | before Nov 17 fixes | relative gate (Nov 17) | + 400 ms lead-in (Nov 18, current) |
|---|---|---|---|
| clean dev | 79.8-89.8 | 86.5-90.1 (R ~0.80, S 1.00) | **94.6** (R 0.93, S 1.00) |
| caller -12 dB | 74.4 (2 misses: -20 dB gate blocked them) | 89.4 | **93.5** (R 0.90, S 1.00) |
| loudstart+8 (stress) | — | 86.2, 2 false stops | **89.7**, 1 false stop (BG-D01: loud early TV = reference) |
- 400 ms lead-in (`PAD_MS`): Whisper turned short abrupt clips of real speech into «ترجمة نانسي قنقر»; with 0.4 s of audio before the burst: stock phrase on 0/27 stop pieces (was 6/27), 6/9 stops caught at the first 0.4 s decode, 0/48 false stops (`prompt_probe.py`). GI-D02 1.20 -> 0.54 s late, AM-D02 1.44 -> 0.51 s. Rejected in the same test: Whisper initial prompt (echoed the prompt as speech -> false STOP), beam-5 retry (swapped one invented phrase for another).
- Older rows below were measured before the lead-in:

| set | before Nov 17 fixes | now (before lead-in) |
|---|---|---|
| caller -6 / +6 dB | — | 87.5-88.6 / 89.9 |
| loudstart+8 (stress) | — | 86.2 (2 false stops: loud TV early becomes the reference) |
| click (stress) | 80.4 (click pushed the reference ~9 dB up) | 80.2 with the sustained reference: the reference is now right (AM-D02 +5 dB vs caller, as on clean), the miss is AM-D02 landing just past the 1.5 s deadline = a late-stop problem; clean 87.0 in the same session |
- Shelved, with numbers: LLM STOP/KEEP judge on undecided texts (2-4 would-be false stops per model, ~1 helpful, 0.5-1 s); cloud audio models as ASR (best 0.83 match in 2.4-3.9 s vs Whisper 0.75 in 0.5 s). Late stops come from short first decodes and the 3-word rule, not from the recogniser.

## Evening of Nov 18 (Day 4)
- **Team dataset** `data/SARJ_dataset_v3` (git-ignored; 58 short calls = 28 scenarios + phone copies + 2 echo; ElevenLabs
  agent voices, one recorded voice per caller line, labels csv-derived, never listened to). `convert_v3.py` ->
  `data/v3_bench` (benchmark layout; evident_ms = onset_ms because it is not labelled, so latency is HARSH there; event
  kinds forced to "interruption"; dialect names mapped). Whisper label check: text match 0.91, no clip below 0.5, so the
  labels are trustworthy. Use it for stop/keep decisions, not latency, and do not tune numbers on it (28 scenarios).
- Base turbo on it (live, before the cue fixes): Core 65.7, R 0.47, **S 1.00** (no false stop on 30 KEEP calls incl. echo,
  TV, cafe, horn, door). Real misses: IH05 «استنى» heard «ستنة/سنة», IH04 phone (stock phrase on the first decodes of very
  short calls), AM02 «طيب؟» (rising intonation, text cannot tell).
- **Arabic-dialect Whisper** `oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2` (Apache 2.0, same turbo size/speed),
  converted to CTranslate2 float16 in `models/turbo-ar-dialectal-v2-ct2` (git-ignored; preprocessor_config.json copied from
  base turbo, 128 mels). Use: `$env:ASR_MODEL="models/turbo-ar-dialectal-v2-ct2"`. `model_compare.py` = offline A/B.
  It writes Gulf letters («وڨف/وگف» for «وقف») -> `normalize()` now maps گ/ڨ->ق, ڤ->ف. Live: team data 70.1 (R 0.54, S 1.00,
  2 misses) vs base 65.7; **dev 88.8 (false stop on BG-D01 TV, miss GI-D03 «لحظة» heard «لحفة»)** vs base 94.9.
  **Decision: keep base turbo** (dev is generated like the hidden set; a false stop is the costliest error), port the
  lessons into rules: «ستنى/ستنا/ستنه» cue variants; near-spelling cue matches need the same first letter («وأني»~«ثواني»
  and «أستاذ»~«استنا» were false cues); «أستاذ/أستاذة» excluded outright. Replay of all logged decodes: only those two
  wrong stops lost, only real cues gained.
- **Current system (base turbo + all Nov 18 fixes), live:** dev **95.0** (R 0.94, S 1.00, best so far; run `cues-dev`),
  team data **71.0** (R 0.53, S 1.00; run `cues-v3_bench`) vs 65.7 before and 70.1 with the dialect model. Remaining team
  misses: AM02 ×2 (intonation), IH04 phone (stock phrase on first decodes of a very short call), IH05 noisy-door phone.
- Deadline news (Nov 18 evening, confirmed): **code freeze AND final submission moved to Saturday Nov 21, 12:00**.
- End of turn: "caller spoke" uses the adaptive floor (fixed -42 dB counted room noise in a noisy place as speech), and
  the end check reads only audio after the agent stopped (the agent's own voice in the mic was read as the caller).
- Demo: Haiku writes the agent's lines with an Arabic Saudi-style prompt; GPT-audio-mini READS them via "your next line is
  already decided: say exactly «…»" (5/5 faithful; the "text-to-speech engine" framing was 1/6, it answered questions).
  `DEMO_WRITER=voice` = old one-step. Speakers/headphones switch (echo cancellation for group demos). `--system v1` lets
  another system file decide (mentor's `my_system_v1.py`, untracked, kept out of git).
- Mentors' suggestion to move work to cloud sessions: declined (all testing needs this laptop's GPU, mic and local data).

## STATUS NIGHT OF NOV 19-20 (Day 5) — START HERE (supersedes the sections below where they disagree)
- **Dates (Sarj, Nov 19 evening): code freeze FRIDAY Nov 20, 8:00 PM (commit hash + service list); report + presentation
  SATURDAY Nov 21, 12:00; final run Saturday.** The older "freeze Saturday 12:00" below is out of date.
- **Focus: only the scored harness stream (80 % Core). The free-conversation demo is frozen: do not work on it.** Keep the
  recorded-call demo working. No overfitting: no per-call fixes; every change must hold on dev, vol-12, loudstart+8, team data
  (`data/v3_bench`) and the polite set (`data/polite`) before it goes into `my_system.py` / `turn_rules.py`.
- **Practice checkpoint (Nov 19, 16 unseen calls; file `sarj-challenge_checkpoint.md`): we are "Pearl", 5th, Core 77.0.**
  Stops when it should 0.63 (5 of 7 interruptions; the misses were one overlap call and one dialect call), keeps talking 0.92
  (2 false stops, both in ONE background-speech call: TV-like talk at the caller's level), median stop 0 ms, p90 1291 ms.
  Top of the board: 91.9 (R 0.87, S 0.96) and 90.1 (R 0.97, S 0.91). Lesson: our weakness is catching interruptions, not
  restraint; latency is not the problem (0.3 s target dropped).
- **What the checkpoint decode log showed** (`asr_log.txt` held only those 16 calls then): 8 of 39 bursts were entirely Whisper
  stock phrases («ترجمة نانسي قنقر», «اشتركوا في القناة»), some loud and 0.9-1.5 s long: probable missed stops.
- **Tested and DROPPED (data in this session):** not counting grammar particles in the 3-word rule (loses 41 of 127 logged real
  stops); a speaker-embedding veto against background voices (`speaker_gate.py`, WeSpeaker ResNet34-LM in `models/`): callers
  have only ~1 s of clean speech before the first interruption, so same-speaker (0.04-0.49) overlaps background (0.05-0.10);
  longer Whisper lead-in (0.8/1.2 s) for stock phrases (helps the first 300 ms only: 30 % -> 19 %, nothing at 600 ms);
  voicing (pitch) as a speech-vs-noise cue (backchannels 0.96, TV 0.74, human speech 0.74: no separation); «يا جماعة» as side
  talk (customers say it to a company too).
- **Being evaluated now: `my_system_next.py` + `turn_rules_next.py` (copies; the frozen files are untouched):**
  1. Second opinion: when the main model returns a stock phrase on a burst, the dialect model
     (`models/turbo-ar-dialectal-v2-ct2`) re-reads it. On 120 real-voice bursts the main model gave a stock phrase 29 % / 9 % / 4 %
     of the time for the first 300 / 600 / 1000 ms; the dialect model 0 %. Switch off: `$env:ASR2_MODEL="off"`.
  2. Polite «لا» (mentor's example «لا لا إن شاء الله» is KEEP): continuation phrases cancel the «لا» cue (`POLITE_PHRASES`),
     and a bare «لا»/«لا لا» stop waits for the next decode while the caller is still talking (`BARE_LA_WAIT=off` disables).
  - Test sets: `data/polite` (made by `make_polite.py`, 7 calls, 9 events, ElevenLabs clips, labels NOT listener-validated).
  - Queue: dev, polite, vol-12, loudstart+8, v3_bench, real time (~1 h). Targets to match: dev 95.0, vol-12 93.5,
    loudstart 89.7, v3_bench 71.0. Track it: `python progress_next.py`.
  - Adopt only if nothing gets worse; then copy the `_next` changes into `my_system.py` / `turn_rules.py`, re-run
    `python -m ttbench.ready`, commit with a CLEAN tree, hand in the hash. Ablate each change (ASR2_MODEL=off, BARE_LA_WAIT=off).
- **RESULTS OF THE CANDIDATES (Nov 20, 00:20): nothing beat the frozen system on held-out data, so the plan is to FREEZE the
  current scored files (commit d12ca5b = my_system.py / turn_rules.py unchanged).** Frozen vs candidates, Core (R, S):
  dev 95.0 | next 94.1 | quiet 93.5 | 93.2 | loud-start 89.7 | 90.9 | team data 71.0 (0.53, 1.00) | next 66.4 (hold hurt: stops
  0.3-0.6 s late) | v2 70.7 (0.53, 1.00) | NEW 38 team-v4 calls (held out) frozen 54.1 (0.39, 0.87) | v2 52.9 (0.38, 0.87) |
  polite set next 82.7 | v2 79.3 (the polite veto alone cannot work: the first decode hears «لا لا» before «إن شاء الله»).
  v2 = `my_system_v2.py` + `turn_rules_v2.py` (repeated-word cue, polite veto, dialect second opinion, no hold).
- **Inspection (56 genuine STOP events of dev + team + team-v4, `inspect_misses.py`):** 45 % stop within 1 s; 41 % later than 1 s
  (mostly first-decode timing; the team sets count latency from the first syllable, so they look worse than the real scorer);
  9 % never stop = ambiguous words «طيب»/«خلاص»/«نعم؟» (text cannot decide; the real scorer treats ambiguous words as "either");
  5 % never stop = Whisper mis-read a repeated «استنى». False stops on keep events are background speech at the caller's level
  (checkpoint BG call, T4-BG01/02, dev BG-D01): no text rule separates it; loudness cannot (the TV is as loud as the caller).
- **Also tested and rejected (all offline, numbers in the session):** ElevenLabs live as the engine (first text 0.75 s, none
  before 0.7 s; same recall as Whisper, slower); acoustic "reflex" from pitch/voicing/spectral features (AUC 0.54, chance);
  speaker veto (enrolment too short); 2-word / 4-word rules; lone question words (Whisper invents «شنو؟» on noise: 6 keep events
  vs 1 stop); "stock phrase on a sustained burst = STOP" (0 of 56 stops, 10-12 keep events have it); lead-in 0.8/1.2 s.
- **Still open, low priority:** RESUME (half credit for a stop taken back within 500 ms); robustness: Whisper never retried
  after 3 failed decodes; a loud warning when Whisper is missing.
- **New team data (teammates are generating it): never tune on it.** Convert it, run it once per candidate change as a held-out
  check, keep one block untouched until the final report.
- **Ops:** start the server with the venv ACTIVE (`(.venv)` in the prompt), otherwise Whisper is missing and the system silently
  falls back to the energy rule (86.1 on dev instead of 95.0; that happened once). Check for `[asr] ... ready` and, with the
  new file, `[asr] second opinion (dialect model) ready`. No `demo.py` or other GPU job during a scored run.
- Known robustness gap: after 3 failed decodes the system never retries Whisper (stays on the energy rule). Fix before the final.

## Status and plan (afternoon of Nov 19) — OLDER, superseded by the section above where they disagree
- Everything is committed (b09ad9b). Untracked on purpose: `my_system_v1.py` (mentor's), `eot_v2_candidate/`.
- Scored system unchanged since Nov 18: dev 95.0, vol-12 93.5, loudstart 89.7, team data 71.0, no false stops.
- Docs current: WRITEUP.md (~2,600 words; failure case 8 = soft «يوم الثلوث» over a question), team plan doc with a
  Nov 19 section + current-pipeline diagram.
- OPEN QUESTION for the user: is the hidden scored stream on Day 6 (Nov 20) or at the Saturday submission? If Nov 20,
  the system must be final tonight.
- Next, in order (agreed list; the user picks):
  1. Freeze dry run: `python -m ttbench.ready` (commit hash + service list), fix anything it flags.
  2. «تقيقة» check: replay every logged decode (asr_log.txt + demo_log.txt) with ت accepted for د in near-spelling cues;
     adopt only if 0 new false stops.
  3. RESUME safety net (take back a wrong stop when the burst turns out to be only an ack/noise within 0.5 s): scored
     change, needs real-time runs on clean + vol-12 + loudstart.
  4. WRITEUP.md: teammates' names (section 9), trim to ~4 pages.
  5. Demo rehearsal on the presenting laptop (plugged in, its own mic, venue network if possible).
  6. Optional demo speed: synthesize the draft's first sentence early (~0.5 s faster replies).

## Known weak spots / next work (priority)
1. **Late stops**: mostly fixed by the 400 ms lead-in (GI-D02/AM-D02 now ~0.5 s). Left: OV-D03 ~0.8 s (3-word rule: «يا أخي هذا» = 2 content words at 0.6 s), GI-D03 ~0.46 s. Idea: 2 words enough when the burst is as loud as the caller's voice. Check on clean + volume + stress sets, not by eye.
2. RESUME: take back a wrong stop within 0.5 s (0.5 credit instead of 0) — insurance for the hidden set.
3. TV / loud background early in a call can become the caller reference (loudstart stress). Text cannot solve BG-D01-type calls; state it in the write-up.
4. Lexicon: 60 multi-word Saudi/Gulf stop phrases added Nov 18 (`PHRASE_CUES`: «ماني فاهم», «وش تقول», «ما يناسبني», «غلط», «عندي سؤال», «عيد الكلام»…), from general usage, not the dev transcripts; replay of 2,719 logged decodes + 171 live demo decodes: 0 new stops (so no dev-score change; it is for unseen callers). Rejected: a 2-word rule when the burst is >= the caller's level (fixes OV-D03 but false-stops BG-D01 TV / NV-D03 in the replay). Still open: «أبغى…» intents (ambiguous with yes-answers), keep collecting from live demo tests.
5. HT-D02 (dictation) takes the turn early; `eot_v2_candidate/` targets it (end-of-turn only, low priority).
- Rule: no thresholds picked by eye from dev points. A number must come from a principle with a wide margin, or survive an augmentation built to break it. The -6 dB line WAS read off dev data (stated honestly in the write-up); -12 dB has a 10 dB margin.

## Demo (`python demo.py`, open http://127.0.0.1:8780 in Edge/Chrome)
- Runs the same `MySystem` on the browser microphone (raw: echo cancellation / AGC off; use headphones). Shows each Whisper decode, decision, reason and the gate lines live.
- Modes: recorded dev call (real agent sentences) or **free conversation**: TAKE_TURN -> Whisper reads the whole turn -> **GPT-audio-mini writes AND speaks the reply in one streamed call** (OpenRouter, ~$0.001/reply, first audio ~1.3-2.2 s), transcript shown; STOP cuts the voice and cancels the stream. Fallback: Claude Haiku 5.5 text + browser voice. Two-step Haiku -> GPT-audio-mini reading was dropped: the voice model *answered* the text instead of reading it. Microsoft neural voices were rejected by the user (robotic).
- **Night of Nov 18, demo rebuilt for speed** (scored system untouched): `.env` also holds `ELEVENLABS_API_KEY` (free tier,
  10k chars/month; key needs TTS + STT access, Voices read) and `GEMINI_API_KEY` (free tier was overloaded/503: not used).
  Page options (free call): **Your words** = ElevenLabs live `scribe_v3_live` (default; streamed while the caller talks,
  committed at 0.3 s silence, final text in ~0.25 s, best accuracy) / batch Scribe v1 / local Whisper / Gemini via
  OpenRouter. **Writer** = Haiku (default, natural) / Gemini 2.5 Flash-Lite (~0.3 s faster, sloppier). **Voice** =
  ElevenLabs built-in Sarah/Chris with `eleven_v4_turbo` (Nov 19; was `eleven_v3_conversational`, `$env:DEMO_ELEVEN_MODEL` switches back) (first sound ~0.5 s; Voice Library Saudi voices need a
  paid plan via API) / GPT-audio-mini voices (fallback). The writer streams sentences, ElevenLabs speaks each as it
  arrives (v3 rejects `previous_text`: only the first piece was spoken until fixed). An early **draft** of the reply starts
  at 0.3 s of caller silence and is thrown away if the caller makes a sound again. Each reply logs a "reply timing" line.
  Measured: ~2.0-2.2 s of silence before the agent speaks with live Scribe (was ~3 s with GPT voice).
  The page's 5 s "no reply" timer is ignored once the caller has spoken (it made the agent talk over a long turn).
  On speakers, live Scribe also hears the agent: clauses of 4+ words that are >=70 % the agent's last line are dropped.
- **Demo UI (Nov 19)**: judge-facing layout. Hero status (READY / AGENT SPEAKING / LISTENING / STOPPED / THINKING, EN+AR),
  current agent sentence + the caller's live words, chat bubbles (live dashed bubble while the caller talks; cut-off agent
  lines marked "you interrupted"; "replied after X s" badges), a 30 s turn-taking timeline (agent speech, caller level,
  STOP / TAKE_TURN markers), call numbers. Settings and the engineer log (meter + table) are collapsed `<details>`.
  Arabic inside English text goes in `<bdi dir="rtl">` (mixed() helper), or it renders scrambled.
  Theme button in the header: auto (Windows setting) / light / dark, remembered per browser.
- Echo in live Scribe (Nov 19): on speakers the agent's lines (even short ones «هلا والله», «يناسبك؟») landed in the
  caller's bubbles. Now `ScribeLive.feed(frame, hold, burst_frames)` sends silence while the agent talks and the engine
  has not stopped it; on STOP (or the agent ending mid-burst) the caller's burst is sent from its start.
  The clause filter stays as a second layer. Release = the STOP's burst even if it already ended (+400 ms before):
  Whisper's STOP often lands after the caller finished («دقيقة معلش، خلّي يوم الثلاثاء» was lost, then replayed
  word for word from `demo_recordings/call-002228.wav`).
- Writer prompt (Nov 19): straight to the point, at most one courtesy phrase, usually one sentence < 12 words, short
  goodbye, name Sarah (it said «فلانة»). Replies went from ~30 to 7-10 words. Voice menu adds "quick speech" voices:
  Flash v2.5 at speed 1.02 (`ELEVEN_STYLE`; 1.05-1.2 were too quick by ear); v4 Turbo and v3 ignore `voice_settings.speed`.
- Seen in that call (scored system, NOT changed): Whisper wrote «تقيقة» for «دقيقة» at 1.3 s; the same-first-letter
  rule rejected it and the STOP came at 1.7 s. A ت/د allowance would need a replay of all logged decodes first.
- Mic lesson: the Razer BlackShark V2 path had a noise gate (16-22 % exact digital silence, chopped words); the laptop mic had 0 %. For Day 6 use the laptop mic or a plain USB mic and test it on the demo machine.

## Write-up (WRITEUP.md, due by the freeze) must declare
- Final system: `mobiuslabsgmbh/faster-whisper-large-v3-turbo` via faster-whisper (local). No cloud call in the scored path.
- Demo only: ElevenLabs Scribe live/v1 (caller words), ElevenLabs TTS v4 Turbo / Flash v2.5 (voice), OpenRouter
  `anthropic/claude-haiku-5.5` (writer), Gemini 2.5 Flash-Lite / Flash (options), `openai/gpt-audio-mini` (fallback voice).
  WRITEUP.md section 8 already lists these (updated Nov 19).
- Experiments: `Systran/faster-whisper-small` (early probes); OpenRouter Claude Haiku 5.5 / Llama 3.3 70B / GPT-4.1 mini (LLM judge); Gemini 2.5/3.7 Flash, Qwen 3.8 omni flash, GPT-audio-mini, Voxtral small (ASR comparison).
- Failure cases with Arabic examples: «اصبر» missing cue (found live), quiet callers vs the old -20 dB gate, Whisper stock phrases, TV as reference, mic noise gate.

## Team docs
- Team plan (Claude Doc): https://claude.ai/artifact/DMXU1CVZyCBUJCeu2sTkPV
- Older summaries: English https://claude.ai/artifact/9W63HZNcw2Cw3QkpS48H8U, Arabic https://claude.ai/artifact/CfBYHPM5zgBachwK8Qzu5K (out of date since Nov 17).

## NOV 20 AFTERNOON (supersedes "freeze d12ca5b" above)
- New scored candidate in `my_system.py` / `turn_rules.py` (uncommitted until the user commits): B speculative STOP on
  2 loud words + RESUME (`SPEC_STOP=off` disables), C dialect cues + Gulf ض/ظ + phonetic near-spelling, F slow-decode
  watchdog (median of 5 decodes > 600 ms -> energy rule). Rejected A (int8, small model, 200 ms first decode), burst
  merging, stock stripping, confidence gates. Full numbers: `analysis/REPORT_nov20.md`; write-up section 5c.
- Live: dev 95.7, val 65.0, train 64.2, LOCKED TEST (run once) 74.7 vs frozen 72.2. Do not tune on data/split_test.
- `sim.py` = deterministic virtual-clock replay (decode cache in analysis/sim_cache); `analysis/variant.py NAME ATTR=V`
  runs variants; splits from `make_splits.py`. Tests: test_dialect_rules.py, test_spec_resume.py, test_asr_robustness.py.
