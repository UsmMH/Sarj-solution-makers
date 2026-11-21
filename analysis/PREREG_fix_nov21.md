# Pre-registration: last-hour fixes (Nov 21, 13:10, BEFORE any run). Hand-in stays 3461916 unless this passes.
Trigger: team set v6 (new) scored 29.5 / noisy copies 53.5 in sim. Variants in my_system_fix.py (sim only):
Q = lone question word with «؟» at >= -3 dB vs the caller -> STOP. W = «لا» + only "go on" words -> keep; bare «لا» waits
one more decode of the burst. Run: Q+W, Q only, W only.
Sets (sim.py, deterministic): 8 dev copies (check fold), polite, team val, team train, new_v6, new_aug.
Pass (vs the current rules, same sim): check-fold mean >= current - 0.5; no copy more than 2 below; false stops on the 8
copies + polite + val + train not above current; new_v6 + new_aug Core not lower. Caveat written now: new_v6/new_aug are
the data that motivated the change, so a gain there is expected and is NOT evidence on its own.

## Result (13:20, nothing changed after the runs)
| variant | 8 dev copies (sim, all 24 calls) | polite | val | train | new_v6 | new_aug |
|---|---|---|---|---|---|---|
| current rules | 97.7/96.3/91.0/97.7/93.8/92.7/90.9/88.2 | 79.3 | 65.4 | 64.9 | 29.5 | 53.5 |
| Q+W | 97.4/96.0/90.7/97.4/93.3/92.4/90.4/87.6 | 72.7 | 65.4 | 63.3 | 58.1 | 53.0 |
| Q only | identical to current, same false stops | 79.3 | 65.4 | 64.9 | **58.6** | 53.5 |
W FAILS (every dev copy lower, polite 79.3 -> 72.7: the wait made real stops late and saved no polite «لا»: Whisper heard
«كمّل» too late). Q PASSES: it never fires on the 11 earlier sets; replay of all logged decodes: it fires on «نعم؟» in
team ambiguous-question calls (rising = the STOP version) and «ها؟» in live demo tests only. Q adopted into turn_rules.py
(sim of the real my_system: identical to the trial on all 13 sets). Caveat: new_v6 motivated it.
