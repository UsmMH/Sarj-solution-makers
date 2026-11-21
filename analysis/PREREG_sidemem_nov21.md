# Pre-registration: side-talk memory (Nov 21, 13:40, BEFORE any run). Hand-in stays 068bbb0 unless this passes.
Seen in a new sample of team noisy calls (T4-NV06: «ماما» = side talk, then «موسيقى» -> default stop = false stop).
Variant (my_system_ds.py, sim only): after a side-talk verdict in a burst, later "too little to tell" decodes of that burst
stay keep (blocks the default and speculative stop there). Cues, phrases, questions and 3+ words still stop.
Sets: 8 dev copies, polite, team val, team train, new_v6, new_aug, new_aug2 (sim.py). Pass: every one of the 8 dev copies,
polite, val, train equal or better with no new false stop; new sets not lower. new_aug2 motivated it (not evidence alone).

## Result (13:50, nothing changed after the run)
Identical on dev + 7 copies, polite, val, train, new_v6, new_aug (same Core, same false stops). new_aug2: 49.4 -> 50.1, false
stops 13 -> 11 (T4-NV06 echo copy: «ماما» then «موسيقى» no longer default-stops). The other NV06 copy is not fixed (its
default stop came before Whisper heard «ماما»). PASS -> adopted in my_system.py; sim of the real file = the trial on all 14.
