"""Offline check of llm_judge.py: replay the real Whisper decodes in asr_log.txt through the LLM.

Only texts that turn_rules.worth_second_opinion() would send are asked (once per scenario). For each one we print
the call's label (STOP / KEEP / EITHER) and the LLM's answer. An LLM STOP in a KEEP call is a would-be false stop;
in a STOP call it is a possible earlier stop (the log has no timestamps, so "earlier" is not checked here).

    python llm_replay.py                                  # default model
    python llm_replay.py --models anthropic/claude-haiku-4.5 google/gemini-2.5-flash
    python llm_replay.py --quiet                          # also ask about quiet bursts

The log holds no agent text, so the LLM gets none ("unknown"): a harder test than the live system.
"""
import argparse
import csv
import re
import statistics
import sys

from llm_judge import LlmJudge
from turn_rules import worth_second_opinion

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LINE = re.compile(r"^\[asr\] burst \d+ .*?: (?P<text>'.*'|\".*\") -> (?:STOP|keep) \((?P<why>.*)\)$")


def scenario_order():
    ids, labels = [], {}
    with open("data/dev/events.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["scenario_id"]
            if sid not in labels:
                ids.append(sid)
                labels[sid] = set()
            labels[sid].add(row["expected"] + (f"({row['preferred']})" if row["preferred"] else ""))
    return ids, labels


def candidates(log_path, quiet):
    ids, labels = scenario_order()
    out, idx, seen = [], -1, set()
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("--- scenario"):
                idx += 1
                continue
            m = LINE.match(line)
            if not m or idx < 0:
                continue
            text, why = eval(m["text"]), m["why"]             # the log writes Python reprs of the text
            sid = ids[idx] if idx < len(ids) else f"#{idx}"
            if worth_second_opinion(text, why, allow_quiet=quiet) and (sid, text) not in seen:
                seen.add((sid, text))
                out.append((sid, "/".join(sorted(labels.get(sid, {"?"}))), text, why))
    if idx + 1 != len(ids):
        print(f"warning: log has {idx + 1} scenarios, labels have {len(ids)}: mapping may be off")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="asr_log.txt")
    ap.add_argument("--models", nargs="+", default=[None])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cands = candidates(args.log, args.quiet)
    print(f"{len(cands)} texts would be sent to the LLM\n")
    for model in args.models:
        judge = LlmJudge(model=model) if model else LlmJudge()
        if not judge.ok:
            print("no OPENROUTER_API_KEY (or LLM=off): nothing to do")
            return
        ok, ms = judge.warm()
        print(f"=== {judge.model}   warm-up: {'ok' if ok else 'FAILED'} ({ms:.0f} ms)")
        times, bad, good = [], 0, 0
        for sid, label, text, why in cands:
            verdict, ms = judge.ask(text)
            times.append(ms)
            flag = ""
            if verdict == "STOP" and "STOP" not in label and "EITHER" not in label:
                bad, flag = bad + 1, "   <-- would be a FALSE STOP"
            elif verdict == "STOP" and "STOP" in label:
                good, flag = good + 1, "   (stop call: may help)"
            print(f"{sid:7} {label:16} {verdict or 'none':5} {ms:5.0f} ms  {text!r}  [{why}]{flag}")
        if times:
            q = sorted(times)
            print(f"--- {judge.model}: median {statistics.median(q):.0f} ms, p90 {q[int(0.9 * (len(q) - 1))]:.0f} ms, "
                  f"max {q[-1]:.0f} ms; STOP in KEEP calls: {bad}; STOP in STOP calls: {good}\n")


if __name__ == "__main__":
    main()
