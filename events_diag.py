import json, glob, wave
import numpy as np

def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) / 32768 + 1e-12)

print(f"{'event':12} {'exp':7} {'dur':>5} {'mean':>6} {'peak':>6} {'p-floor':>7} {'centHz':>7}  text")
rows = []
for lp in sorted(glob.glob("data/dev/labels/*.json")):
    lab = json.load(open(lp, encoding="utf-8"))
    sid = lab["scenario_id"]
    w = wave.open(f"data/dev/audio/{sid}.wav")
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    ca = d[:, 1].astype(np.float64)
    n = len(ca) // 320
    lv = np.array([db(ca[i*320:(i+1)*320]) for i in range(n)])
    floor = np.percentile(lv, 10)
    texts = {s["seg_id"]: s.get("text", s.get("kind", "")) for s in lab["user_segments"]}
    for e in lab["events"]:
        if "onset_ms" not in e or "offset_ms" not in e:
            continue
        a = int(e["onset_ms"] / 20)
        b = max(a + 1, int(e["offset_ms"] / 20))
        seg = lv[a:b]
        s = ca[a*320:b*320]
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        cent = (np.fft.rfftfreq(len(s), 1/16000) * spec).sum() / (spec.sum() + 1e-9)
        txt = " ".join(texts.get(x, "") for x in e.get("segment_ids", []))
        rows.append((e["expected"], e["event_id"], e["offset_ms"] - e["onset_ms"],
                     seg.mean(), seg.max(), seg.max() - floor, cent, txt))
for r in sorted(rows):
    print(f"{r[1]:12} {r[0]:7} {r[2]:5d} {r[3]:6.1f} {r[4]:6.1f} {r[5]:7.1f} {r[6]:7.0f}  {r[7]}")