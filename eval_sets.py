"""No-overfit evaluation on the 24 provided calls only (no team data).

Folds (fixed before any result, by call number, so every one of the 8 situations is in both):
    inspect = the 8 *-D01 calls   : the ONLY calls we may listen to / read logs of to find failures
    check   = the 16 *-D02, *-D03 : never inspected call by call; a change must not lose here
Sets: the dev calls plus label-safe copies where only the caller channel changes (augment.py):
    dev, vol-12, loudstart+8, click, phone, noise-pink15, noise-brown5, reverb

Acceptance rule for a change (decided now, before any result):
    1. check-fold Core, averaged over all 8 sets, goes up;
    2. no set loses more than 2 points on the check fold (live runs vary about +-2);
    3. no new false stop on any set (S never drops on check);
    4. the change was motivated by an inspect-fold call or a general principle, never by a check-fold call.

    python eval_sets.py run --system my_system_eleven.py --tag el-base          # all 8 sets, real time, ~85 min
    python eval_sets.py run --system my_system_eleven.py --tag el-base --sets phone,reverb
    python eval_sets.py table el-base [el-try1 ...]                              # scores per set and fold
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from ttbench.score import aggregate, load_labels, score

SETS = {"dev": "data/dev", "vol-12": "data/aug/vol-12", "loudstart+8": "data/aug/loudstart+8", "click": "data/aug/click",
        "phone": "data/aug/phone", "noise-pink15": "data/aug/noise-pink15", "noise-brown5": "data/aug/noise-brown5",
        "reverb": "data/aug/reverb"}
EXTRA = {"polite": "data/polite", "val": "data/split_val", "train": "data/split_train"}   # run only, not in the fold table
FOLDS = {"inspect": lambda sid: sid.endswith("-D01"), "check": lambda sid: not sid.endswith("-D01"),
         "all": lambda sid: True}


def run_name(tag, s):
    return f"{tag}__{s}"


def cmd_run(a):
    sets = a.sets.split(",") if a.sets else list(SETS)
    for s in sets:
        name = run_name(a.tag, s)
        data = SETS.get(s) or EXTRA[s]
        print(f"=== {name} ({data})", flush=True)
        r = subprocess.run([sys.executable, "-m", "ttbench.play", "--system", a.system, "--calls", "all",
                            "--data", data, "--name", name], capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = [l for l in r.stdout.splitlines() if "Core" in l or "core" in l][-2:]
        print("\n".join(tail) or r.stdout[-400:], flush=True)
        if r.returncode != 0:
            print(r.stderr[-800:], flush=True)


def fold_scores(run_dir: Path, data_dir: Path) -> dict:
    labels, manifest = load_labels(data_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    log = [json.loads(l) for l in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for fold, keep in FOLDS.items():
        m = dict(meta, scenarios=[s for s in meta["scenarios"] if keep(s["scenario_id"])])
        res = score({k: v for k, v in labels.items() if keep(k)}, m, log, manifest.get("thresholds"))
        out[fold] = {"core": res["core"]["core_score"], "R": res["core"]["R"], "S": res["core"]["S"],
                     "fs": res["rates"]["false_stops"], "miss": res["rates"]["misses"],
                     "p50": res["latency_ms"]["median"], "p90": res["latency_ms"]["p90"]}
    return out


def cmd_table(a):
    rows = {}
    for tag in a.tags:
        for s, d in SETS.items():
            rd = Path("runs") / run_name(tag, s)
            if (rd / "score.json").exists():                    # finished runs only
                rows[(tag, s)] = fold_scores(rd, Path(d))
    for fold in ("check", "inspect", "all"):
        print(f"\n== {fold} fold ({'16 calls' if fold == 'check' else '8 calls' if fold == 'inspect' else '24 calls'}): "
              f"Core (R / S, false stops, misses, stop latency p50/p90 ms)")
        print(f"{'set':14}" + "".join(f"{t:>44}" for t in a.tags))
        means = {t: [] for t in a.tags}
        for s in SETS:
            line = f"{s:14}"
            for t in a.tags:
                r = rows.get((t, s), {}).get(fold)
                if r is None:
                    line += f"{'-':>44}"
                    continue
                means[t].append(r["core"])
                lat = f"{r['p50'] or 0:.0f}/{r['p90'] or 0:.0f}"
                line += f"{r['core']:7.1f} ({r['R']:.2f}/{r['S']:.2f}, fs {r['fs']}, miss {r['miss']}, {lat})".rjust(44)
            print(line)
        print(f"{'MEAN':14}" + "".join(f"{(sum(v) / len(v) if v else 0):7.1f} over {len(v)} sets".rjust(44)
                                       for v in means.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--sets", default="")
    t = sub.add_parser("table")
    t.add_argument("tags", nargs="+")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    main()
