#!/usr/bin/env python3
"""Fit the image->ground homography from measured points and write it into a profile.

1. Put >= 4 small markers on the floor in front of the car; measure each one
   from the REAR AXLE centre: x = metres to the right, y = metres forward.
2. Grab a frame (tools/stream.py --raw, or a frame from a log) at the profile's
   camera resolution, and read off each marker's pixel (u, v).
3. Run:
   python tools/ground_cal.py profiles/x.yaml \
       --pt 112,170:-0.20,0.50 --pt 205,171:0.20,0.50 \
       --pt 135,128:-0.20,1.00 --pt 186,128:0.20,1.00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from racer import profile  # noqa: E402
from racer.geometry import GroundMapper, fit_homography  # noqa: E402


def parse_pt(s: str):
    px, g = s.split(":")
    u, v = map(float, px.split(","))
    x, y = map(float, g.split(","))
    return (u, v), (x, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--pt", action="append", required=True, metavar="U,V:X,Y")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args()

    prof = profile.load(args.profile)
    pairs = [parse_pt(s) for s in args.pt]
    if len(pairs) < 4:
        sys.exit("need at least 4 points")
    px = np.array([p for p, _ in pairs])
    gr = np.array([g for _, g in pairs])
    w, h = prof.camera.width, prof.camera.height
    H = fit_homography(px, gr, prof.vision.lens, w, h)
    m = GroundMapper(prof.vision.lens, prof.vision.geometry, w, h, H.tolist())
    err = np.linalg.norm(m.pixels_to_ground(px) - gr, axis=1)
    for (p, g), e in zip(pairs, err):
        print(f"  px {p} -> ground {g}: error {e * 100:.1f} cm")
    print(f"mean error {err.mean() * 100:.1f} cm")
    if not args.dry_run:
        prof.vision.ground_homography = H.tolist()
        profile.save(prof, args.profile)
        print(f"wrote ground_homography to {args.profile}")


if __name__ == "__main__":
    main()
