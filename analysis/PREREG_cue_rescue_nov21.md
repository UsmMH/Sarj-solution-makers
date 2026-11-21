# Pre-registration: cue-only rescue (written Nov 20 ~23:55, BEFORE any run)

Base: submitted hybrid `2bc041c` (`my_system_hybrid.py`). Variant: `my_system_hybrid_cue.py`, one change to the rescue:
1. ElevenLabs may STOP only when decide() on its text returns a cue or a phrase (never "N words", never "question").
   Reason: 12 of 13 live rescues were cues; the 13th was TV news reaching the 3-word rule. Fluent sentences are where
   background speech lives, and Whisper already handles them (it is not "blind" on long clear speech).
2. A bare «لا» cue from ElevenLabs is judged only on its final text of the burst, and never when the rest is a polite
   ending («لا لا إن شاء الله», «لا الله يعافيك»; list as turn_rules_next.py). Reason: the mentors' KEEP example.
Whisper's own decisions are unchanged (so the polite «لا» KEEP is NOT fixed on the Whisper side; this only makes sure the
rescue does not add such stops).

No other variant, no setting changed after seeing results.

## Runs (live, real time; queue started overnight)
A. variant on the 8 sets (check fold = 16 `*-D02/D03` calls).
B. hybrid `2bc041c` repeated on the 8 sets (how much of +1.3 was noise).
C. base `my_system.py`, hybrid, variant on team data split_val (11) + split_train (44): never used to set numbers; includes
   background-speech calls (TV risk on other voices). split_test stays locked.
D. base, hybrid, variant on data/polite (7 calls; labels not listener-validated).

## Acceptance (variant replaces the hybrid only if ALL hold)
1. 8-set check-fold mean >= the hybrid's mean (average of its two runs) - 0.5 (it is a restriction: equal is the aim);
2. no set more than 2 points below the hybrid (two-run average);
3. false stops <= hybrid on the 8 sets, on team data, and on the polite set.
If the variant has fewer false stops on team data or polite with 1 and 2 holding, it is adopted.
The hybrid stays the hand-in if the hybrid adds false stops on team data vs base: report it, and the user decides
(fall back to `my_system.py` = a3e0bd0 behaviour, or the variant).

## Result (Nov 21, 03:40; queue finished, nothing re-run or changed)
Check fold (16 calls), Core per set: dev / vol-12 / loudstart+8 / click / phone / pink15 / brown5 / reverb = mean
- hybrid run 1: 96.3 / 94.6 / 92.5 / 96.5 / 94.5 / 93.5 / 94.9 / 89.7 = 94.0
- hybrid run 2: 97.4 / 96.3 / 93.0 / 97.4 / 93.9 / 94.0 / 94.9 / 90.9 = 94.7   (two-run average 94.35; run noise ~0.7)
- cue-only:     97.5 / 96.3 / 93.0 / 97.4 / 93.8 / 97.0 / 95.0 / 90.5 = 95.1
1. mean 95.1 >= 94.35 - 0.5: holds.
2. worst set vs the hybrid average: phone -0.4 (limit -2): holds.
3. false stops: check fold 1 vs 1 / 1 (phone BG-D03, the same Whisper stop in all runs); team val 0 / 0 / 0 and train
   3 / 3 / 3 KEEP losses (base / hybrid / cue); polite 5 / 5 / 5: holds.
   Team data, base vs hybrid: same false stops (val 65.5 / 65.5, train 65.7 / 64.9), so the hybrid's own clause is clear.
VERDICT: PASS -> cue-only replaces the hybrid as the hand-in.
Honest reading: the cue variant can only remove rescue STOPs, so its +0.75 over the hybrid average is run noise (pink
+3.0 is Whisper timing, not the rescue). The real reason to switch is safety: in these runs the full hybrid made 3
non-cue rescues, two of them TV news («في الشأن المحلي، أعلنت», «في الشغل المحلي») that cue-only cannot make; all 14
cue-only rescues were real cues (لحظة ×5, بس ×5, عفوا ×2, استنى, اسمع), none a bare «لا». Polite «لا» stays a Whisper-side loss
(S 0.67 on data/polite in all three systems). Inspect fold (not used to decide): 93.4 / 94.3 / 93.2.
