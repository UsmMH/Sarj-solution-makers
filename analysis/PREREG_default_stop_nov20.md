# Pre-registration: refined "default stop" (written Nov 20 ~19:50, BEFORE any run of these variants)

Base: submitted Whisper system (926a2af), simulated (`sim-base__*`, check-fold mean 90.1).
Trial v1 (`sim-ds__*`, 1 non-backchannel word, 1 s, -6 dB): check mean 90.1 = tie; new false stop OV-D02 on 4 of 8 sets. Rejected.
We did NOT look at OV-D02's audio or text (check fold). The variants below come from principles only.

## Variants (exactly two, fixed now)
- **DS-2W**: STOP when the burst is >= 1.0 s, peak >= -6 dB vs the caller's reference voice, and the text has
  **>= 2 non-backchannel content words**. Principle: one stray word is what a recogniser makes of noise, side speech or a
  polite filler; two content words at the caller's own level over a full second is a phrase aimed at the agent.
- **DS-3DB**: STOP when the burst is >= 1.0 s, peak **>= -3 dB** vs the caller's reference voice, and >= 1 content word.
  Principle: the -3 dB line is the existing "as loud as the caller's own voice" line used by the speculative STOP
  (Nov 20); background or side speech is usually quieter than the caller's own mouth-to-mic voice.
No other values will be tried. If both fail, default stop is dropped.

## Acceptance rule (same as the protocol)
On the CHECK fold (16 `*-D02/D03` calls), simulated, 8 sets (dev, vol-12, loudstart+8, click, phone, noise-pink15,
noise-brown5, reverb), against `sim-base`:
1. mean Core over the 8 sets goes up;
2. no set loses more than 2 points;
3. no new false stop: per set, false stops <= base;
then a live real-time confirmation on dev + the two sets with the largest simulated change before it can be submitted.
If both variants pass, the one with the higher check mean wins; ties go to DS-2W (stricter = safer).
