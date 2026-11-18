# Data request for the team's generator (paste this whole file into Claude)

We are building a real-time turn-taking system for an Arabic (Saudi/Gulf) phone agent. Every 20 ms it decides whether
the agent should STOP talking because the caller wants the floor, or KEEP talking (backchannel, noise, echo, hesitation).
We need **evaluation calls** to test it on voices and phrasings it has never seen. The calls are an exam, not training
data: precise labels matter more than quantity. **Deadline: before Nov 19, 12:00** (code freeze is 18:00).

## What to make (in this order of priority)

1. **More caller voices.** For each caller line, 3 to 5 different speakers (male and female, young and older). Never only
   one voice per line.
2. **Interruptions without an obvious stop word** (expected STOP):
   corrections and complaints («يا أخي هذا ثالث مرة أسمع نفس الكلام», «مو هذا اللي طلبته», «الرقم غلط»),
   questions («كم المبلغ؟», «متى يوصل؟», «ليش كذا؟»), and interruptions that START like a backchannel
   («إيه بس أبي أسأل…», «طيب لا، لحظة…», «تمام بس الموعد ما يناسبني»).
3. **Tone pairs: same words, different intent.** «طيب» falling and calm (KEEP: go on) vs «طيب؟» rising (STOP: a question);
   «خلاص» calm (KEEP: OK) vs «خلاص خلاص!» firm (STOP). Same speaker for both members of a pair.
4. **Background speech as loud as the caller** (expected KEEP): a TV or radio presenter, or another person in the room
   talking near the phone, while the caller is silent. Also some calls where that background is present AND the caller
   then interrupts (STOP).
5. **Non-speech sounds from the caller** (expected KEEP): cough, laugh, sneeze, heavy breath, throat clearing, a child
   shouting in the room.
6. **Other Gulf dialects:** Emirati, Bahraini, Qatari, Kuwaiti, more Hijazi, in any of the categories above.

Do NOT make: volume, noise or telephone-line copies of existing calls (we generate those ourselves).

## Audio format (exactly like the organisers' dev set)

* WAV, 16 kHz, 16-bit PCM, **stereo**: channel 1 (left) = agent only, channel 2 (right) = caller + caller's background.
* Agent speech about -21 dBFS RMS, peaks at or below -6 dBFS. Caller at natural levels (soft speakers softer).
* Continuous room tone (about -62 dBFS) on the caller channel for the whole call: never exact digital silence.
* Calls of 15 to 30 s, ideally with 2 to 3 agent sentences (the agent speaks first, the caller answers, the agent
  continues, and the event happens during a later agent sentence), so the caller's normal voice is heard before the event.
* The agent keeps talking through the whole interruption in the audio (the system under test decides to stop; the audio
  must not stop by itself).

## Labels: one JSON per call, this structure

```json
{
  "schema_version": "1.0",
  "scenario_id": "TQ-N01",
  "split": "team",
  "category": "GI",
  "dialect": "sa-najdi",
  "elicitation": "scripted",
  "speakers": [{"speaker_id": "T001", "role": "caller", "source": "tts or human", "gender": "m", "age_band": "25-34", "dialect": "sa-najdi"}],
  "acoustic": {"conditions": ["quiet"], "snr_db": 40},
  "audio": {"file": "TQ-N01.wav", "sample_rate": 16000, "channels": ["agent", "user"], "duration_ms": 21000},
  "context": {"use_case": "appointment reminder", "language": "ar"},
  "agent_utterances": [
    {"utt_id": "A0", "start_ms": 300, "end_ms": 4500, "text": "السلام عليكم، معك ريم من العيادة. معي الأستاذ فهد؟", "is_question": true},
    {"utt_id": "A1", "start_ms": 5900, "end_ms": 19800, "text": "حبيت أذكرك بموعدك يوم الأحد الساعة عشرة الصبح…", "is_question": false}
  ],
  "user_segments": [
    {"seg_id": "U0", "speaker_id": "T001", "start_ms": 4800, "end_ms": 5400, "kind": "speech", "text": "إيه نعم"},
    {"seg_id": "U1", "speaker_id": "T001", "start_ms": 9200, "end_ms": 11800, "kind": "speech", "text": "عفواً، بس الأحد ما يناسبني", "delivery": "polite"}
  ],
  "background_events": [],
  "events": [
    {"event_id": "TQ-N01.e1", "kind": "interruption", "expected": "STOP", "segment_ids": ["U1"],
     "onset_ms": 9200, "evident_ms": 9750, "offset_ms": 11800, "rationale": "polite interruption, intent clear at «بس»"}
  ]
}
```

Rules for the labels:
* `category`: GI interruption, BC backchannel, HT hesitation, BG background, NV non-verbal, AM ambiguous, OV overlap, DV dialect.
* `events[].kind` is always `"interruption"` for STOP and KEEP events (also for noise and backchannels).
* `expected`: `STOP`, `KEEP`, or `EITHER` (then add `"preferred": "STOP"` or `"KEEP"`).
* **`evident_ms` is required for every STOP and EITHER event**: the moment the intent becomes clear, usually the START of
  the cue word or of the first word that makes it a floor bid («بس» in «عفواً، بس أبغى…», «لحظة» in «ممم… لحظة»). Measure
  it on the final WAV, not from the script.
* `onset_ms` = when the caller's sound starts; `offset_ms` = when it ends. Measure on the final WAV.
* `text` must be what is really said in the audio. If a voice changed a word, fix the text.
* KEEP events also need `onset_ms` and `offset_ms`. Non-speech sounds: `"kind": "nonverbal"` in `user_segments`, text like «(كحة)».

Deliver as a folder: `labels/*.json`, `audio/*.wav`, and a `manifest.json`:
`{"dataset": "team-v4", "split": "team", "scenarios": [{"scenario_id": "TQ-N01"}, ...]}`.
Please also list, in a short README, which voices/speakers were used for which calls, and anything you could not check
(for example: nobody listened to file X).
