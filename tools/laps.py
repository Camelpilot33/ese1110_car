#!/usr/bin/env python3
"""Lap times of a run, grouped by speed settings.

  python tools/laps.py logs/<run>            # table per (speed_max, speed_min)
  python tools/laps.py logs/<run> --laps     # every lap
  python tools/laps.py logs/<run> --json out.json

New runs log a "lap" event per A press (duration, settings, timed/settle/
restart). Older runs are reconstructed from the A button presses and the
settings logged with every button event; a lap counts only if the car drove
the whole lap (no dead-man stop) and the settings did not change during it.
Laps outside 0.85x-1.15x the median (mis-timed presses; normal spread is ~3 %) are flagged, not counted.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml


def load_laps(run: Path) -> list:
    S = [json.loads(l) for l in open(run / "state.jsonl")]
    E = [json.loads(l) for l in open(run / "events.jsonl")]
    t0 = S[0]["t"] if S else 0.0
    lap_ev = [e for e in E if e.get("event") == "lap"]
    laps = []
    if lap_ev:
        for e in lap_ev:
            laps.append(dict(t=e["t"] - t0, duration=e["duration"], kind=e["kind"],
                             speed_max=e.get("speed_max"), speed_min=e.get("speed_min")))
    else:
        prof = yaml.safe_load(open(run / "profile.yaml"))
        cur = (prof["control"].get("speed_max"), prof["control"].get("speed_min"))
        timeline = []
        for e in E:
            if e.get("button"):
                timeline.append((e["t"], (e.get("speed_max", cur[0]), e.get("speed_min", cur[1]))))

        def setting_at(t):
            s = cur
            for ts, v in timeline:
                if ts <= t:
                    s = v
            return s

        a = [e["t"] for e in E if e.get("button") == "a"]
        ok_t = [(s["t"], s["ok"]) for s in S]
        for t1, t2 in zip(a, a[1:]):
            stopped = any(not ok for t, ok in ok_t if t1 <= t < t2)
            s1, s2 = setting_at(t1 + 1e-6), setting_at(t2 - 1e-6)
            kind = "restart" if stopped else ("changed" if s1 != s2 else "timed")
            laps.append(dict(t=t1 - t0, duration=t2 - t1, kind=kind, speed_max=s1[0], speed_min=s1[1]))
    timed = [l["duration"] for l in laps if l["kind"] == "timed"]
    if timed:
        med = statistics.median(timed)
        for l in laps:
            if l["kind"] == "timed" and not (0.85 * med <= l["duration"] <= 1.15 * med):
                l["kind"] = "mistimed"
    return laps


def summarise(laps: list) -> list:
    groups = {}
    for l in laps:
        if l["kind"] == "timed":
            groups.setdefault((l["speed_max"], l["speed_min"]), []).append(l["duration"])
    rows = []
    for (mx, mn), d in sorted(groups.items()):
        rows.append(dict(speed_max=mx, speed_min=mn, n=len(d), mean=statistics.mean(d), best=min(d),
                         sd=statistics.pstdev(d) if len(d) > 1 else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--laps", action="store_true", help="list every lap")
    ap.add_argument("--json", help="write laps + summary to this file")
    args = ap.parse_args()
    laps = load_laps(Path(args.run))
    if not laps:
        sys.exit("no laps (press A at the start line every lap)")
    if args.laps:
        for i, l in enumerate(laps, 1):
            print(f"{i:3d}  t={l['t']:6.1f}s  {l['duration']:6.3f}s  max {l['speed_max']}  min {l['speed_min']}  {l['kind']}")
    rows = summarise(laps)
    counts = {k: sum(1 for l in laps if l["kind"] == k) for k in {l["kind"] for l in laps}}
    print(f"{len(laps)} laps: {counts}")
    print(f"{'speed_max':>9} {'speed_min':>9} {'n':>3} {'mean':>7} {'best':>7} {'sd':>6}")
    for r in rows:
        print(f"{r['speed_max']:>9} {r['speed_min']:>9} {r['n']:>3} {r['mean']:7.3f} {r['best']:7.3f} {r['sd']:6.3f}")
    if args.json:
        json.dump({"laps": laps, "summary": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
