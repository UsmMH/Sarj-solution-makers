"""Offline check of turn_rules.decide against the ASR probe transcripts. No audio, no GPU, takes a second.

    python eval_rules.py                                       # every asr_probe_*.txt in this folder
    python eval_rules.py asr_probe_large-v3-turbo_cuda-float16.txt

For each labelled event it applies the rule to the first-600-ms transcript (PRE) and to the whole-burst
transcript (FULL). The live system will decode the growing burst again and again, so an event counts as
STOP if EITHER snapshot says stop. Full table goes to eval_rules_out.txt (UTF-8): paste that file back.
"""
import glob
import json
import re
import sys
from pathlib import Path

from turn_rules import decide

ROOT = Path(__file__).resolve().parent
LINE = re.compile(r"^(\S+) \| (\w+) \| (.*?) \| PRE: (.*) \| FULL: (.*)$")
TAIL = re.compile(r"\s*\([\d.]+, \d+ ms\)\s*$")


def heard(s: str) -> str:
    s = TAIL.sub("", s).strip()
    return "" if s == "-" else s


def agent_text_for(event_id: str) -> str:
    """The agent sentence being spoken when the event starts (for the echo check); '' if unknown."""
    p = ROOT / "data" / "dev" / "labels" / f"{event_id.split('.')[0]}.json"
    if not p.exists():
        return ""
    lab = json.loads(p.read_text(encoding="utf-8"))
    onset = next((e.get("onset_ms") for e in lab["events"] if e["event_id"] == event_id), None)
    if onset is None:
        return ""
    return next((u["text"] for u in lab["agent_utterances"] if u["start_ms"] <= onset <= u["end_ms"]), "")


def check(path: Path) -> list:
    out = [f"=== {path.name}", ""]
    stops = keeps = stops_early = stops_ok = keeps_ok = 0
    wrong = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        eid, exp, said, pre, full = m.groups()
        agent = agent_text_for(eid)
        (s1, r1), (s2, r2) = decide(heard(pre), agent), decide(heard(full), agent)
        stop = s1 or s2
        ok = exp == "EITHER" or (stop if exp == "STOP" else not stop)
        if exp == "STOP":
            stops += 1
            stops_ok += stop
            stops_early += s1
        elif exp == "KEEP":
            keeps += 1
            keeps_ok += not stop
        row = (f"{'ok   ' if ok else 'WRONG'} {eid:10} {exp:6} said={said} | PRE: {'STOP' if s1 else 'keep'} "
               f"({r1}) | FULL: {'STOP' if s2 else 'keep'} ({r2})")
        out.append(row)
        if not ok:
            wrong.append(row)
    out += ["", f"STOP events caught: {stops_ok}/{stops}  (already at 600 ms: {stops_early})",
            f"KEEP events kept:   {keeps_ok}/{keeps}", ""]
    return out


def main():
    files = [Path(a) for a in sys.argv[1:]] or [Path(p) for p in sorted(glob.glob(str(ROOT / "asr_probe_*.txt")))]
    if not files:
        sys.exit("No asr_probe_*.txt files found next to this script.")
    lines = []
    for f in files:
        lines += check(f)
    text = "\n".join(lines)
    (ROOT / "eval_rules_out.txt").write_text(text, encoding="utf-8")
    for ln in lines:
        if ln.startswith(("===", "STOP events", "KEEP events", "WRONG")):
            print(ln)
    print("\nFull table in eval_rules_out.txt: paste its contents back.")


if __name__ == "__main__":
    main()