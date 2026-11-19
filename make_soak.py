"""Build one long stream for a soak test: the 24 dev calls three times (clean, caller -12 dB, caller +6 dB) = 72 calls,
about 30 minutes in real time, like the hidden stream (64-80 scenarios).

    python make_soak.py
    python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --data data/soak --name soak

Scenario ids get a suffix (GI-D01 -> GI-D01-a / -b / -c) so they stay unique. Labels are unchanged otherwise.
"""
import hashlib
import json
import shutil
from pathlib import Path

PARTS = [("a", Path("data/dev")), ("b", Path("data/aug/vol-12")), ("c", Path("data/aug/vol+6"))]
OUT = Path("data/soak")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    (OUT / "labels").mkdir()
    base = json.loads((PARTS[0][1] / "manifest.json").read_text(encoding="utf-8"))
    manifest = dict(base, split="soak", scenarios=[])
    event_lines, header = [], None
    for tag, src in PARTS:
        m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        ids = [s["scenario_id"] for s in m["scenarios"]]
        for sid in ids:
            new = f"{sid}-{tag}"
            text = (src / "labels" / f"{sid}.json").read_text(encoding="utf-8").replace(sid, new)
            label = json.loads(text)
            old_wav = src / "audio" / json.loads((src / "labels" / f"{sid}.json").read_text(encoding="utf-8"))["audio"]["file"]
            new_wav_name = f"{new}.wav"
            shutil.copy2(old_wav, OUT / "audio" / new_wav_name)
            label["audio"]["file"] = new_wav_name
            lab_path = OUT / "labels" / f"{new}.json"
            lab_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["scenarios"].append({"scenario_id": new, "wav_sha256": sha(OUT / "audio" / new_wav_name),
                                          "label_sha256": sha(lab_path)})
        lines = (src / "events.csv").read_text(encoding="utf-8").splitlines()
        header = lines[0]
        for ln in lines[1:]:
            sid = ln.split(",")[0]
            event_lines.append(ln.replace(sid, f"{sid}-{tag}"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "events.csv").write_text("\n".join([header] + event_lines) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(manifest['scenarios'])} calls")


if __name__ == "__main__":
    main()
