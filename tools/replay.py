#!/usr/bin/env python3
"""Run vision + control on a logged run (laptop or Pi). No hardware needed.

  python tools/replay.py logs/20261107_190000_manual
  python tools/replay.py logs/<run> --profile profiles/x.yaml --set control.lookahead_m=0.8
  python tools/replay.py logs/<run> --video out.mp4      # annotated video
  python tools/replay.py logs/<run> --show               # window (laptop)

Uses the run's own profile.yaml unless --profile is given. Prints detection
rate, steering statistics and per-frame processing time; with manual logs it
also compares auto steering with what the driver did.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer import profile  # noqa: E402
from racer.camera import LogSource  # noqa: E402
from racer.control import Controller  # noqa: E402
from racer.vision import Vision, draw_debug  # noqa: E402


def load_states(run: Path) -> dict:
    p = run / "state.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        s = json.loads(line)
        out[s["frame"]] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--profile")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--video")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0, help="playback fps for --video/--show")
    args = ap.parse_args()

    run = Path(args.run)
    prof = profile.load(args.profile or run / "profile.yaml", args.set)
    vision = Vision(prof.vision, prof.camera.width, prof.camera.height)
    ctrl = Controller(prof.control)
    states = load_states(run)
    src = LogSource(run)

    writer = None
    n = valid = 0
    proc_ms, steer, diffs, phases = [], [], [], {"straight": 0, "turn": 0}
    for i, (idx, img) in enumerate(src):
        st = states.get(idx, {})
        now = st.get("t", i / args.fps)
        t0 = time.perf_counter()
        est = vision.process(img, keep_mask=bool(args.video or args.show))
        cmd = ctrl.update(est, now)
        proc_ms.append((time.perf_counter() - t0) * 1000)
        n += 1
        valid += est.valid
        if est.valid:
            steer.append(cmd.steering)
            phases[cmd.phase] += 1
            if st.get("mode") == "manual" and st.get("ok"):
                diffs.append(cmd.steering - st["steering"])
        if args.video or args.show:
            txt = f"#{idx} {cmd.phase} s={cmd.steering:+.2f} t={cmd.throttle:+.2f} k={cmd.kappa_preview:+.2f}"
            if "steering" in st:
                txt += f" log_s={st['steering']:+.2f}"
            vis = draw_debug(img, vision, est, cmd.target, txt)
            vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            if args.video:
                if writer is None:
                    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.show:
                cv2.imshow("replay", vis)
                k = cv2.waitKey(int(1000 / args.fps)) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    cv2.waitKey(0)
    if writer:
        writer.release()

    print(f"frames {n}   valid {valid} ({100 * valid / max(n, 1):.1f} %)")
    print(f"processing ms: mean {statistics.mean(proc_ms):.2f}  p95 {np.percentile(proc_ms, 95):.2f}")
    if steer:
        print(f"steering: mean {statistics.mean(steer):+.3f}  std {statistics.pstdev(steer):.3f}  "
              f"|max| {max(map(abs, steer)):.2f}   phases {phases}")
    if diffs:
        print(f"auto - manual steering: mean {statistics.mean(diffs):+.3f}  "
              f"rms {float(np.sqrt(np.mean(np.square(diffs)))):.3f}  (n={len(diffs)})")
    loop = [s["loop_ms"] for s in states.values() if "loop_ms" in s]
    if loop:
        print(f"logged on-car loop ms: mean {statistics.mean(loop):.1f}  max {max(loop):.1f}")
    ev = run / "events.jsonl"
    if ev.exists():
        laps = [json.loads(l)["t"] for l in ev.open() if '"button": "a"' in l]
        if len(laps) > 1:
            lt = np.diff(laps)
            print(f"laps {len(lt)}: best {lt.min():.3f}  mean {lt.mean():.3f}  "
                  f"mean of last 5 {lt[-5:].mean():.3f}")


if __name__ == "__main__":
    main()
