"""Convert data/SARJ_dataset_v3 (the team's generated calls) into the benchmark layout, so ttbench can play it.

    python convert_v3.py                 # -> data/v3_bench/{manifest.json, labels/, audio/}
    python -m ttbench.play --url ws://127.0.0.1:8765 --calls all --data data/v3_bench --name v3

What is filled in (the source labels do not have it):
  * scenario_id = the call's file name (AM01_clean, AM01_clean_phone, ...), unique per call;
  * context.use_case = the category (the harness sends it to the system);
  * evident_ms = onset_ms for STOP / EITHER events. The real moment the intent becomes clear («بس» in
    «عفواً، بس أبغى…») was never labelled, so latency here is measured from the first syllable: HARSHER than the
    dev set. Use this set for stop/keep decisions, not for latency.
The audio is copied unchanged.
"""
import json
import shutil
from pathlib import Path

SRC = Path("data/SARJ_dataset_v3")
DST = Path("data/v3_bench")


def main():
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "labels").mkdir(parents=True)
    (DST / "audio").mkdir()
    dev = json.loads(Path("data/dev/manifest.json").read_text(encoding="utf-8"))
    scenarios = []
    for ann in sorted((SRC / "annotations").glob("*.json")):
        d = json.loads(ann.read_text(encoding="utf-8"))
        sid = ann.stem
        d["scenario_id"] = sid
        d["split"] = "v3"
        d.setdefault("context", {})["use_case"] = d.get("category", "call")
        # the scorer's slice names: "sa-najdi" for Najdi; noise-only calls have no caller, so no dialect
        d["dialect"] = {"Najdi": "sa-najdi"}.get(d.get("dialect"), f"sa-{d['dialect'].lower()}" if d.get("dialect") else "none")
        conds = d.setdefault("acoustic", {}).setdefault("conditions", [])
        d["acoustic"]["conditions"] = ["quiet" if c in ("clean", "room_tone") else c for c in conds] or ["quiet"]
        for ev in d.get("events", []):
            if ev.get("expected") in ("STOP", "EITHER") and "evident_ms" not in ev:
                ev["evident_ms"] = ev["onset_ms"]
            if ev.get("expected") in ("STOP", "KEEP", "EITHER"):
                ev["kind"] = "interruption"     # the scorer's word for every stop/keep event (dev uses it for noise too);
                                                # "background"/"echo" made it treat them as end-of-turn events
        (DST / "labels" / f"{sid}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copy2(SRC / "calls" / d["audio"]["file"], DST / "audio" / d["audio"]["file"])
        scenarios.append({"scenario_id": sid})
    manifest = {"dataset": "sarj-team-v3", "version": "3", "split": "v3", "thresholds": dev["thresholds"],
                "label_status": "csv_derived; evident_ms = onset_ms (not labelled)", "scenarios": scenarios}
    (DST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{DST}: {len(scenarios)} calls")


if __name__ == "__main__":
    main()
