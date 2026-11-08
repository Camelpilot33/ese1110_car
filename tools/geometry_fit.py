#!/usr/bin/env python3
"""Self-calibrate the camera pitch/height from a logged run, using the fact
that the two tape lines are parallel with a constant width.

  python tools/geometry_fit.py logs/<run> --profile profiles/x.yaml --width 0.40 [--write]

1. detects both lines on sampled frames with the profile's current geometry
2. converts those points back to pixels (so the result does not depend on
   the starting guess)
3. searches pitch_deg so the measured width is the same at every distance
4. sets height_m so the width equals --width (the real lane width between
   tape centres; measure it with a tape measure)

Does nothing when ground_homography is set (that comes from measured points).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer import profile  # noqa: E402
from racer.geometry import GroundMapper  # noqa: E402
from racer.vision import Vision  # noqa: E402


def collect_pairs(run: Path, prof, step: int, max_frames: int):
    """Pixel coordinates of (left, right) points that lie on the same image row."""
    v = Vision(prof.vision, prof.camera.width, prof.camera.height)
    states = [json.loads(l) for l in open(run / "state.jsonl")] if (run / "state.jsonl").exists() else []
    frames = [s["frame"] for s in states if s.get("ok", True)] or \
        [int(p.stem) for p in sorted((run / "frames").glob("*.jpg"))]
    pairs = []
    for fr in frames[::step][:max_frames]:
        img = cv2.imread(str(run / "frames" / f"{fr:06d}.jpg"))
        if img is None:
            continue
        e = v.process(img)
        if len(e.left_pts) < 3 or len(e.right_pts) < 3:
            continue
        lp = v.mapper.ground_to_pixels(e.left_pts)
        rp = v.mapper.ground_to_pixels(e.right_pts)
        rows = {int(round(p[1])): p for p in rp if np.isfinite(p).all()}
        for p in lp:
            if np.isfinite(p).all() and int(round(p[1])) in rows:
                pairs.append((p, rows[int(round(p[1]))]))
    return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])


def widths(vcfg, geo, w, h, lp, rp):
    m = GroundMapper(vcfg.lens, geo, w, h)
    gl, gr = m.pixels_to_ground(lp), m.pixels_to_ground(rp)
    ok = np.isfinite(gl).all(1) & np.isfinite(gr).all(1) & (gl[:, 1] > 0) & (gl[:, 1] < vcfg.max_range_m * 2)
    return gr[ok, 0] - gl[ok, 0], (gl[ok, 1] + gr[ok, 1]) / 2


def spread(width, y):
    """Relative change of width with distance (slope of width/mean vs y) + scatter."""
    if len(width) < 50:
        return np.inf
    a, b = np.polyfit(y, width, 1)
    return abs(a) / np.mean(width) + 0.2 * np.std(width) / np.mean(width)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--width", type=float, required=True, help="real lane width between tape centres (m)")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--write", action="store_true", help="update the profile")
    args = ap.parse_args()

    prof = profile.load(args.profile)
    if prof.vision.ground_homography is not None:
        sys.exit("profile has a measured ground_homography; geometry fit not needed")
    w, h = prof.camera.width, prof.camera.height
    lp, rp = collect_pairs(Path(args.run), prof, args.step, args.max_frames)
    print(f"{len(lp)} left/right point pairs")
    if len(lp) < 100:
        sys.exit("not enough frames with both lines visible")

    geo0 = prof.vision.geometry
    wd, y = widths(prof.vision, geo0, w, h, lp, rp)
    print(f"before: pitch {geo0.pitch_deg:.1f} deg, width {np.percentile(wd, 10):.3f}..{np.percentile(wd, 90):.3f}, "
          f"slope {np.polyfit(y, wd, 1)[0]:+.3f} m per m")

    best = None
    for pitch in np.arange(-10.0, 45.0, 0.25):
        geo = dataclasses.replace(geo0, pitch_deg=float(pitch))
        s = spread(*widths(prof.vision, geo, w, h, lp, rp))
        if best is None or s < best[0]:
            best = (s, float(pitch))
    geo = dataclasses.replace(geo0, pitch_deg=best[1])
    wd, y = widths(prof.vision, geo, w, h, lp, rp)
    # Ground coordinates scale linearly with camera height (x_offset is small).
    geo = dataclasses.replace(geo, height_m=round(geo0.height_m * args.width / float(np.median(wd)), 4),
                              pitch_deg=round(best[1], 2))
    wd, y = widths(prof.vision, geo, w, h, lp, rp)
    print(f"after:  pitch {geo.pitch_deg:.2f} deg, height {geo.height_m:.3f} m, width "
          f"{np.percentile(wd, 10):.3f}..{np.percentile(wd, 90):.3f}, slope {np.polyfit(y, wd, 1)[0]:+.3f} m per m")
    for lo in np.arange(0.0, 1.6, 0.2):
        m = (y >= lo) & (y < lo + 0.2)
        if m.sum() > 20:
            print(f"   y {lo:.1f}-{lo + 0.2:.1f} m: width {np.median(wd[m]):.3f}  (n={m.sum()})")
    if args.write:
        prof.vision.geometry = geo
        prof.vision.track_width_m = args.width
        profile.save(prof, args.profile)
        print(f"wrote geometry + track_width_m to {args.profile}")


if __name__ == "__main__":
    main()
