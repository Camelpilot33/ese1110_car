#!/usr/bin/env python3
"""Venue calibration (~1 minute). Run on the Pi with the car on the track.

  python tools/calibrate.py --venue levine --base profiles/default.yaml

1. auto exposure / white balance settle, then the values are read and locked
2. exposure is capped (--max-exposure-us) to limit motion blur; gain makes up
3. at each of N track positions (press Enter), the tape colour is sampled:
   pixels that differ strongly from the floor/sheet colour inside the ROI and
   are saturated are taken as tape; HSV thresholds = percentiles + margin
4. writes profiles/<date>_<venue>.yaml and saves the sample frames to logs/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer import profile  # noqa: E402


def tape_pixels(bgr: np.ndarray, roi_top: float, roi_bottom: float,
                min_sat: int = 70, min_dist: float = 40.0) -> np.ndarray:
    """HSV values of likely tape pixels in the ROI (Nx3)."""
    h = bgr.shape[0]
    roi = bgr[int(roi_top * h):int(roi_bottom * h)]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    floor = np.median(lab, axis=0)
    dist = np.linalg.norm(lab - floor, axis=1)
    sel = (dist > min_dist) & (hsv[:, 1] >= min_sat)
    return hsv[sel]


def thresholds(samples: np.ndarray, margin=(4, 25, 25)):
    lo = np.percentile(samples, 3, axis=0) - margin
    hi = np.percentile(samples, 97, axis=0) + margin
    lo = np.clip(lo, 0, [179, 255, 255]).astype(int)
    hi = np.clip(hi, 0, [179, 255, 255]).astype(int)
    return lo.tolist(), hi.tolist()


def main():
    from racer.camera import PiCamera

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venue", required=True)
    ap.add_argument("--base", default="profiles/default.yaml")
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--max-exposure-us", type=int, default=8000)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--out", help="output path (default profiles/<date>_<venue>.yaml)")
    args = ap.parse_args()

    prof = profile.load(args.base)
    prof.name = f"{time.strftime('%Y-%m-%d')}_{args.venue}"
    out = Path(args.out or f"profiles/{prof.name}.yaml")
    shots = Path("logs") / f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
    shots.mkdir(parents=True)

    cam = PiCamera(prof.camera, auto=True)
    try:
        print(f"letting auto exposure settle for {args.settle_s:.0f} s ...")
        time.sleep(args.settle_s)
        md = cam.metadata()
        exp, gain = md["ExposureTime"], md["AnalogueGain"]
        cg = list(md["ColourGains"])
        print(f"auto: exposure {exp} us, gain {gain:.2f}, colour gains {cg[0]:.3f}/{cg[1]:.3f}, lux {md.get('Lux', 0):.0f}")
        if exp > args.max_exposure_us:
            gain = min(gain * exp / args.max_exposure_us, 16.0)
            exp = args.max_exposure_us
            print(f"capped: exposure {exp} us, gain {gain:.2f}")
        prof.camera.exposure_us, prof.camera.analogue_gain = int(exp), round(float(gain), 3)
        prof.camera.colour_gains = [round(float(g), 4) for g in cg]
        cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(exp),
                          "AnalogueGain": float(gain), "ColourGains": tuple(cg)})
        time.sleep(0.5)

        samples = []
        for i in range(args.positions):
            input(f"[{i + 1}/{args.positions}] place the car on the track (tape visible), press Enter ")
            f = cam.get(timeout=1.0)
            img = f[0]
            cv2.imwrite(str(shots / f"{i:02d}.jpg"), img)
            px = tape_pixels(img, prof.vision.roi_top, prof.vision.roi_bottom)
            print(f"   {len(px)} tape pixels, median HSV {np.median(px, axis=0).astype(int).tolist() if len(px) else '-'}")
            if len(px):
                samples.append(px)
    finally:
        cam.close()

    if not samples or sum(len(s) for s in samples) < 200:
        print("not enough tape pixels found; keeping thresholds from the base profile")
    else:
        lo, hi = thresholds(np.vstack(samples))
        prof.vision.hsv_lower, prof.vision.hsv_upper = lo, hi
        print(f"tape HSV range: {lo} .. {hi}")

    profile.save(prof, out)
    print(f"wrote {out}\nsample frames in {shots}\n"
          f"check with: python tools/stream.py --profile {out}")


if __name__ == "__main__":
    main()
