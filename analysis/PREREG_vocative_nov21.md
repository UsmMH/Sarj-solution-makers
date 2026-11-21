# Pre-registration: «يا + name» as side talk (Nov 21, 10:25, BEFORE any replay or sim run)

Question from the team: a caller who shouts to someone in the room («يا محمد!», «يا أحمد جيب الشاي») should not cut the
agent. Today: a short call is kept ("too little to tell"), but 3+ words, a question, or 1 s of loud speech STOP.

Change (turn_rules.py + 4 lines in my_system.py):
- «يا» + a common Saudi given name (fixed list of ~65 in analysis/vocative_scan.py, without «علي» (= «على» after
  normalising) and «أمل» (= "hope")) -> "talking to someone else in the room" (keep), which also blocks the speculative
  STOP, the default stop and the rescue (they act only on "too little to tell").
- NOT when the agent said that name anywhere in this call: every dev call opens «معك ريم … معي الأستاذ عمر؟», so «يا ريم»
  is the caller talking to the agent. Whole-call memory, reset per call.
- Checked AFTER stop cues and phrases: «يا محمد لحظة» still stops (a cue is our most reliable signal).
- «يا مدير», «يا أستاذ», «يا أخي» are NOT names: still stop (they can address the agent).

Tests and pass rule:
1. Replay of all logged decodes (asr_log, demo_log, eleven_log; ~20.8k): list every decision that changes; pass only if
   no changed decode is a real interruption of the agent.
2. Deterministic sim (cached decodes) of the check fold on dev + the 7 copies, team val/train, polite: pass only if every
   score is identical (expected: the names never occur in those calls).
3. Unit cases: «يا محمد» / «يا محمد وين المفتاح؟» keep; «يا ريم متى يوصل؟» (agent is ريم) stop; «يا محمد لحظة» stop;
   «واحد فول يا مدير» stop.
Expected score effect on the hidden set: ~0 (no such event in 24 dev calls); the gain is natural behaviour in the live demo.
If any test fails: not adopted, the hand-in stays 1c666c6.

## Result (10:45, nothing changed after the runs)
1. Replay of 20,791 logged decodes: the only changed decisions are 2 STOPs in the team's own live side-talk test this
   morning (demo_log: «يا أحمد دي كيكي», «يا أحمد دي كيك يا أحمد», caller calling someone in the room) -> keep. PASS.
2. Deterministic sim, rule on vs off: identical on all 11 sets (dev 97.7, vol-12 96.3, loudstart 91.0, click 97.7,
   phone 93.8, pink 92.7, brown 90.9, reverb 88.2, polite 79.3, val 65.4, train 64.9; same false stops and misses). PASS.
3. Unit cases: all 11 as intended (incl. «يا ريم متى يوصل الطلب؟» -> STOP when the agent is ريم, «يا محمد لحظة» -> STOP). PASS.
VERDICT: adopted. Score effect on data like the hidden set: none measured (no such event in any set); it makes the live
demo behave naturally when a caller calls someone in the room.
