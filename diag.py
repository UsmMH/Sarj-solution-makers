import wave, glob, os
import numpy as np

def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) / 32768 + 1e-12)

print(f"{'id':8} {'floor':>7} {'median':>7} {'p90':>7} {'%>-42':>6} {'first1s':>8} {'echo r':>7}")
for p in sorted(glob.glob("data/dev/audio/*.wav")):
    w = wave.open(p)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    n = len(d) // 320
    ag = np.array([db(d[i*320:(i+1)*320, 0]) for i in range(n)])
    ca = np.array([db(d[i*320:(i+1)*320, 1]) for i in range(n)])
    r = np.corrcoef(ag, ca)[0, 1]
    print(f"{os.path.basename(p)[:-4]:8} {np.percentile(ca,10):7.1f} {np.median(ca):7.1f} "
          f"{np.percentile(ca,90):7.1f} {100*np.mean(ca>-42):6.0f} {ca[:50].mean():8.1f} {r:7.2f}")