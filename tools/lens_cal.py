#!/usr/bin/env python3
"""Lens distortion from chessboard photos -> profile `vision.lens`.

The car camera is wide-angle with strong barrel distortion, so do this once
per camera (it does not change between venues unless the lens is touched).

1. Print a chessboard (e.g. 9x6 inner corners) and tape it to something flat.
2. Record ~20 frames with the board at different positions/angles:
     python run.py --mode manual --log --dry-run     (move the board around)
3. python tools/lens_cal.py profiles/default.yaml "logs/<run>/frames/*.jpg" --board 9x6
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer import profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("images", help="glob, quoted")
    ap.add_argument("--board", default="9x6", help="inner corners, COLSxROWS")
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cols, rows = map(int, args.board.lower().split("x"))
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    paths = sorted(glob.glob(args.images))
    step = max(1, len(paths) // (args.max_images * 3))
    obj, img_pts, size = [], [], None
    for p in paths[::step]:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if ok:
            c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            obj.append(objp)
            img_pts.append(c)
            if len(obj) >= args.max_images:
                break
    print(f"board found in {len(obj)} images")
    if len(obj) < 8:
        sys.exit("need at least 8 good views")
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img_pts, size, None, None, flags=flags)
    w, h = size
    print(f"rms reprojection error {rms:.3f} px")
    prof = profile.load(args.profile)
    lens = prof.vision.lens
    lens.fx, lens.fy = round(K[0, 0] / w, 5), round(K[1, 1] / h, 5)
    lens.cx, lens.cy = round(K[0, 2] / w, 5), round(K[1, 2] / h, 5)
    lens.k1, lens.k2 = round(float(D[0, 0]), 5), round(float(D[0, 1]), 5)
    print(lens)
    if not args.dry_run:
        profile.save(prof, args.profile)
        print(f"wrote lens to {args.profile}  (re-run tools/ground_cal.py afterwards)")


if __name__ == "__main__":
    main()
