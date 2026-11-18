# CLAUDE.md — Team 4, Sarj Arabic Turn-Taking Hackathon

Read this first. Hand-over from the chat sessions that built the system (Nov 15-18, 2026). Last update: evening of
Nov 18 (Day 4). Code freeze and final submission: **Saturday Nov 21, 12:00** (moved from Nov 19).

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
  has not stopped it; on STOP (or the agent ending mid-burst) the caller's burst is sent from its start (+100 ms).
  The clause filter stays as a second layer.
- Mic lesson: the Razer BlackShark V2 path had a noise gate (16-22 % exact digital silence, chopped words); the laptop mic had 0 %. For Day 6 use the laptop mic or a plain USB mic and test it on the demo machine.

## Write-up (WRITEUP.md, due by the freeze) must declare
- Final system: `mobiuslabsgmbh/faster-whisper-large-v3-turbo` via faster-whisper (local). No cloud call in the scored path.
- Demo only: OpenRouter `openai/gpt-audio-mini` (voice + reply), `anthropic/claude-haiku-5.5` (fallback).
- Experiments: `Systran/faster-whisper-small` (early probes); OpenRouter Claude Haiku 5.5 / Llama 3.3 70B / GPT-4.1 mini (LLM judge); Gemini 2.5/3.7 Flash, Qwen 3.8 omni flash, GPT-audio-mini, Voxtral small (ASR comparison).
- Failure cases with Arabic examples: «اصبر» missing cue (found live), quiet callers vs the old -20 dB gate, Whisper stock phrases, TV as reference, mic noise gate.

## Team docs
- Team plan (Claude Doc): https://claude.ai/artifact/DMXU1CVZyCBUJCeu2sTkPV
- Older summaries: English https://claude.ai/artifact/9W63HZNcw2Cw3QkpS48H8U, Arabic https://claude.ai/artifact/CfBYHPM5zgBachwK8Qzu5K (out of date since Nov 17).
