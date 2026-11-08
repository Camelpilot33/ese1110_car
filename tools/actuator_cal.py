#!/usr/bin/env python3
"""Bench calibration of steering centre/limits and throttle. PUT THE CAR ON A STAND.

  python tools/actuator_cal.py --profile profiles/default.yaml

Type a pulse width in microseconds and press Enter:
  s 1700     steering pulse        t 1520     throttle pulse
  c          centre steering + stop throttle
  q          quit (stops everything)
Write the values you find into the profile's `actuators:` section.
Hints: steering_center_us = pulse where the car rolls straight;
throttle: find the smallest pulse above stop where the wheels turn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racer import profile  # noqa: E402
from racer.actuators import Actuators  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/default.yaml")
    args = ap.parse_args()
    cfg = profile.load(args.profile).actuators
    act = Actuators(cfg)
    lo = min(cfg.throttle_reverse_us, cfg.throttle_forward_us)
    hi = max(cfg.throttle_reverse_us, cfg.throttle_forward_us)
    print(__doc__)
    try:
        while True:
            line = input("> ").strip().split()
            if not line:
                continue
            if line[0] == "q":
                break
            if line[0] == "c":
                act.set(0.0, 0.0)
                continue
            if len(line) != 2 or line[0] not in "st":
                print("?")
                continue
            us = float(line[1])
            if line[0] == "s":
                print("ticks", act.pca.set_pulse_us(cfg.steering_channel, us))
            else:
                if not lo <= us <= hi:
                    print(f"refusing: throttle pulse outside {lo}..{hi} us")
                    continue
                print("ticks", act.pca.set_pulse_us(cfg.throttle_channel, us))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        act.close()


if __name__ == "__main__":
    main()
