# Pre-registration: Whisper + ElevenLabs "rescue" hybrid (written Nov 20 ~20:45, BEFORE any run)

Base: submitted system `a3e0bd0` (Whisper + default stop).

## The one variant (no other settings will be tried)
`my_system_hybrid.py` = the base system, unchanged, plus ElevenLabs `scribe_v3_live` streaming the caller channel.
An ElevenLabs transcript of the open caller burst may trigger STOP only when:
1. turn_rules.decide() on the ElevenLabs text says STOP (cue, phrase, question, 3+ words; all vetoes and loudness gates
   apply with the same burst peak and caller reference voice as Whisper), and
2. Whisper has NOT actively vetoed this burst: its latest decode of the burst is missing, empty, a stock phrase, or
   "acknowledgement or too little to tell". If Whisper said echo / side talk / far quieter / quiet words, ElevenLabs
   cannot overrule it.
No speculative STOP and no default stop from ElevenLabs text (Whisper keeps those). End of turn: Whisper only.
ElevenLabs down: the base system exactly.
Principle: Whisper is faster on clean audio; ElevenLabs is more accurate and never invents subtitle text, and is much
better in room echo (check fold 90.2 vs 81.3). The rescue lets ElevenLabs fill in only where Whisper is blind.

## Acceptance rule (live, real time, CHECK fold = 16 `*-D02/D03` calls, 8 sets)
Against the base system run live on the same 8 sets:
1. mean Core over the 8 sets goes up;
2. no set loses more than 2 points;
3. no new false stop: per set, false stops <= base.
Also reported, not part of the rule: inspect fold, stop latency, ElevenLabs disconnects (must be 0 for a valid run).
Cost and risk to state in the write-up if adopted: needs internet at the venue (without it = the base system).

## Result (live, real time, Nov 20 ~22:45) — PASSES the rule
Check fold Core, base `a3e0bd0` -> hybrid: dev 96.2 -> 96.3, vol-12 94.6 -> 94.6, loudstart+8 89.7 -> 92.5,
click 95.5 -> 96.5, phone 93.2 -> 94.5, pink15 93.7 -> 93.5, brown5 93.6 -> 94.9, reverb 85.2 -> 89.7.
Mean 92.7 -> 94.0; worst set -0.2 (pink); check false stops 1 -> 1 (same phone call). ElevenLabs disconnects: 0.
Inspect fold (not in the rule): 93.5 -> 93.4; OV-D01 missed on brown5 by BOTH systems (noise, not the hybrid).
13 rescue STOPs: 12 are cue words Whisper missed («لحظة», «بس», «اسمع», «لأ»); 1 was TV news text
(«في الشأن المحلي، أعلنت») that caused no new false stop here but shows the risk: ElevenLabs reads background TV well.
Live runs vary ~±2 per set; the gain is consistent across sets (7 of 8 up or level) rather than large on one.
