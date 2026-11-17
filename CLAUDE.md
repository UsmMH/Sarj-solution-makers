# CLAUDE.md — Team 4, Sarj Arabic Turn-Taking Hackathon

Read this first. It is the hand-over from the chat session that built the system so far (Nov 15-17, 2026).

## Who and what
- User: Usama, final-year Information Systems student at KSU, working with a small team (some non-technical). He wants to go **step by step**, and wants **full git commands** (`git add ...` + `git commit -m "..."`), never just a message. He runs on Windows / PowerShell, RTX 3060 Ti.
- Event: Sarj AI Human-Like Turn Taking Hackathon, Arabic phone-call agent. 5 teams. Our system decides, every 20 ms, whether the agent should STOP, RESUME or TAKE_TURN.
- Final ranking = 80 % Core score on a **hidden Arabic stream (Day 6)** + 20 % judges (originality 10, practical applicability 5, clarity 5).
- Day 3 checkpoint: evening of Nov 17, 2026 (Asia/Riyadh). First test on unseen data. Paste its feedback into the chat/analyze it first.
- Team name constant in `my_system.py` must stay: `TEAM = "Team 4 (for now :D)"`. Also keep `PEAK_DB = -20.0`.

## Repo layout (this folder = the git repo, branch `main`)
- `my_system.py` — the system: adaptive-floor energy gate + local faster-whisper ASR worker thread + end-of-turn probe. Never blocks the 20 ms loop.
- `turn_rules.py` — pure-Python Arabic text rules: `decide(text, agent_text, peak_db)` -> (STOP?, reason); `eot_incomplete(text)` for end of turn.
- `eval_rules.py`, `asr_probe.py`, `diag.py`, `events_diag.py` — offline helpers (probe transcripts / rule eval).
- Starter-kit files (do not edit): CHALLENGE/DATA/INTERFACE/README/SCORING/SCRIPTS/SUBMISSION/TROUBLESHOOTING/WRITEUP .md, `ttbench/`, `scenarios/`, `schema/`, `tests/`, `data/`, `examples/`.
- `eot_v2_candidate/` — **untested** next version of the two files (see "End of turn"). Not part of the committed system.
- Do NOT commit: `asr_log.txt`, `asr_probe_*.txt`, `asr_small_cpu.txt`, `eval_rules_out.txt`, `runs/`, `.venv/` (already in `.gitignore`).

## How to run (two PowerShell windows, always real time; `--fast` is invalid with ASR)
```
cd C:\Users\ososh\Work\Sarj_makers\turn-taking
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass      # once per window
.\.venv\Scripts\Activate.ps1
# window 1
python my_system.py                  # loads Whisper, then listens on ws://127.0.0.1:8765
# window 2
python -m ttbench.play --url ws://127.0.0.1:8765 --calls quick   # 8 calls, ~3.5 min
python -m ttbench.play --url ws://127.0.0.1:8765 --calls all     # 24 calls, ~10 min
```
- Env switches: `$env:ASR="off"` (energy-only fallback; **remember to `Remove-Item Env:ASR`** — this has bitten us), `$env:EOT="off"` (old fixed 0.7 s end-of-turn), `$env:ASR_WARM="off"`, `$env:ASR_MODEL="small"`.
- Delete `asr_log.txt` before a run you want to read (it appends across runs). It holds every decode: `[asr] burst ...` for stops, `[eot] ...` for end-of-turn probes, `--- scenario {...}` markers (scenario ids are NOT printed, only use_case; order = GI-D01..DV-D03 as in the play output).
- Packages beyond requirements.txt: `pip install faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`. Without them the system silently falls back to the energy rule.
- Reports: `runs/<name>/report.md`; `runs/history.csv`.

## Scoring (from SCORING.md)
- STOP: full credit if stopped <=300 ms after "evident", linear to 0.4 at 1500 ms, later = miss. KEEP (backchannel/noise/hesitation/echo): stopping = 0. EITHER (ambiguous): preferred 1.0, other 0.6.
- R = mean STOP credit per category; S = KEEP credit x "no stray stop" per category; F = harmonic(R,S); Core = 100*(0.9F + 0.1A).
- End-of-turn score = harmonic(patience, promptness) — **its own leaderboard, NOT in the total / ranking**. Only 2 of 24 dev calls exercise it (HT-D01, HT-D02). Don't over-invest in it.
- Timing thresholds are provisional; `manifest.json` holds the ones in force.

## Pipeline (current, committed)
1. Adaptive noise floor (20th percentile of last 3 s + 10 dB, min -42 dB) flags a caller burst while the agent talks.
2. Worker thread transcribes the growing burst with Whisper large-v3-turbo (float16, CUDA, language ar, beam 1, max_new_tokens 24; ~270 ms/decode; GPU kept warm with silent decodes).
3. `decide()`: stop cues (لحظة, استنى, اسمع, بس, لا at sentence start...), questions, >=3 distinct content words => STOP; acks / echo of the last 3 agent sentences / side-talk / Whisper stock hallucinations («ترجمة نانسي قنقر», «اشتركوا في القناة») => keep. Weak evidence (question, word count) is blocked when burst peak < -20 dB (a loudness cue that holds on the synthetic dev data — may NOT hold on the hidden set; test it).
4. Fallback: if ASR is missing or fails 3 times, energy rule with peak gate (-20 dB).
5. End of turn (committed "smart" version): when the caller is quiet 300 ms, decode their last 4 s; if the text ends mid-thought (trailing «...» / comma, filler اممم, or last word و/يعني/في/...) wait up to 1.6 s instead of 0.7 s; if no ASR answer by 0.9 s take the turn anyway.

## Score history on the 24 dev calls
- Starter baseline 41.0 -> loudness-only 86.1 -> with Whisper + rules: 79.8 to 89.8 (same code varies run to run because Whisper garbles first words, esp. AM-D02).
- Latest committed run: **Core 89.8** (R 0.83, S 1.00), 0 missed, 0 false stops, end-of-turn 65.6 (was 28.5).

## Known weak spots
- GI-D02 stops ~1.1 s late («أيوه… بس»); AM-D02 ~1.1 s late or missed depending on the run; OV-D03 0.7 s late.
- noisy_or_phone slice R ~0.37-0.68 vs ~0.96 clean.
- BG-D01 (TV / other speech): false-stopped in some runs, kept in the last two; text can't solve this class.
- HT-D02 (order-number dictation) still takes the turn early 3 times (digits and «لحظة أشوف» pauses).
- Lexicon gaps: negated understanding («ماني فاهم»), «؟» questions, fillers «يعني»/«شسمه», «أبغى» intents.

## End of turn: v2 candidate (not yet run)
`eot_v2_candidate/my_system.py` and `turn_rules.py` add: wait up to 2.8 s after «لحظة / ثواني / أشوف» near the end of the text, and wait 1.6 s after a trailing string of >=2 number words (digit dictation). Replaying the real `[eot]` transcripts changed decisions only in HT-D02. To test: copy both files over the repo versions, delete `asr_log.txt`, run all 24, compare end-of-turn score (65.6) and Core (89.8 +- run noise). If not better: `git restore my_system.py turn_rules.py`.

## Honest status / rules for this project
- The 24 dev calls were all seen and tuned on => **no honest unseen test exists**. Stop tuning on them. Plan: teammates record Arabic speech, mix over agent audio with exact labels in the `data/dev` JSON format (run with `--data`), split by speaker, leave-one-speaker-out; volume / band-limit / noise perturbations of the 24 calls as a stress test of the loudness rule. A "freeze 6-8 of the 24 as a holdout" rule was discussed but not finalized.
- Declare in the write-up (WRITEUP.md): `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (final), `Systran/faster-whisper-small` (early probes), faster-whisper library; plus any recorded/outside data later. No external APIs.
- Summary reports for teammates/organizers exist as Claude Docs: English https://claude.ai/artifact/9W63HZNcw2Cw3QkpS48H8U and Arabic https://claude.ai/artifact/CfBYHPM5zgBachwK8Qzu5K (keep them in sync when scores change).

## Next steps (priority)
1. Checkpoint feedback (tonight) -> analyze before changing anything.
2. Read BG-D01 lines in `asr_log.txt` to explain its false stop.
3. Build the fair test set (team recordings) + perturbation stress test.
4. Noisy/phone robustness (try Silero VAD / `vad_filter`); Arabic lexicon gaps; late stops (GI-D02).
5. v2 end-of-turn test (low priority, not in ranking).
6. Freeze code a couple of days before Day 6; fill WRITEUP.md; final connection check on the submission machine.
