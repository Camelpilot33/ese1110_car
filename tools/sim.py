#!/usr/bin/env python3
"""Closed-loop oval simulator for offline controller tuning.

Kinematic bicycle on a left-hand oval, with the real racer.control.Controller
in the loop. Vision is replaced by sampling the true centre line inside the
camera's visible range and fitting it the same way vision.py does.

Model constants are fitted to the lab runs (2026-11-07):
  * turn radius 0.67 m (preview curvature in turns), straights 0.47 m
  * speed = 7.0 * (throttle - 0.18) m/s   -> ~3.5 s laps at run-6 throttles
  * lateral grip 3.0 m/s^2: holds at 0.375 turn throttle (run 6),
    runs wide at 0.40 (run 7)
  * 30 ms camera/loop delay, 80 ms servo lag, 1 cm measurement noise
All in the profile's ground units. Treat results as relative (A vs B), not
as lap-time predictions.

  python tools/sim.py --profile profiles/2026-11-07_lab.yaml
  python tools/sim.py --profile profiles/x.yaml --set control.kd=0.4 --set control.kp=3
  python tools/sim.py --profile profiles/x.yaml --grid control.kp=1,2,3 control.kd=0.1,0.25,0.5
"""
from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from racer import profile  # noqa: E402
from racer.control import Controller  # noqa: E402
from racer.vision import TrackEstimate  # noqa: E402

R_TURN, L_STRAIGHT = 0.67, 0.47
V_GAIN, V_OFFSET, V_TAU = 7.0, 0.18, 0.3
A_LAT_MAX = 3.0
UNDERSTEER_K = 0.3           # kappa_eff = kappa_cmd / (1 + K v^2); fits run 3 (15 cm outside at 0.38)
DELAY_S, SERVO_TAU, NOISE_M = 0.03, 0.08, 0.01
DT = 1 / 60
Y_VIS = (0.2, 0.75)          # visible range of the centre line (m ahead)


class Oval:
    """Left-hand (counter-clockwise) oval centre line, parameterised by arc length."""

    def __init__(self, r=R_TURN, ls=L_STRAIGHT):
        self.r, self.ls = r, ls
        self.length = 2 * ls + 2 * math.pi * r

    def point(self, s):
        s %= self.length
        r, ls = self.r, self.ls
        if s < ls:                                    # bottom straight, heading +x
            return np.array([s - ls / 2, -r]), 0.0
        s -= ls
        if s < math.pi * r:                           # right turn (left-hand), centre (ls/2, 0)
            a = -math.pi / 2 + s / r
            return np.array([ls / 2 + r * math.cos(a), r * math.sin(a)]), a + math.pi / 2
        s -= math.pi * r
        if s < ls:                                    # top straight, heading -x
            return np.array([ls / 2 - s, r]), math.pi
        s -= ls
        a = math.pi / 2 + s / r
        return np.array([-ls / 2 + r * math.cos(a), r * math.sin(a)]), a + math.pi / 2

    def project(self, p, s_guess):
        """Arc length of the closest centre-line point near s_guess, and signed
        lateral offset of p (+ = right of travel direction)."""
        ss = s_guess + np.linspace(-0.3, 0.3, 61)
        pts = np.array([self.point(s)[0] for s in ss])
        i = int(np.argmin(np.linalg.norm(pts - p, axis=1)))
        s = ss[i]
        c, h = self.point(s)
        right = np.array([math.sin(h), -math.cos(h)])
        return s % self.length, float(np.dot(p - c, right))


def observe(track, pos, yaw, s_car, rng):
    """Centre-line points in the car frame (x right, y forward) inside the
    visible range, quadratic fit as in vision.py."""
    pts, y_last = [], -1e9
    for ds in np.linspace(0.0, 1.2, 60):
        c, _ = track.point(s_car + ds)
        d = c - pos
        x = d[0] * math.sin(yaw) - d[1] * math.cos(yaw)     # right
        y = d[0] * math.cos(yaw) + d[1] * math.sin(yaw)     # forward
        if y < y_last:
            break       # path curls back toward the car: the row scan stops here
        y_last = y
        if Y_VIS[0] <= y <= Y_VIS[1] and abs(x) < 0.8:
            pts.append((x + rng.normal(0, NOISE_M), y))
    est = TrackEstimate()
    if len(pts) < 4:
        return est
    p = np.array(pts)
    deg = 2 if len(p) >= 6 else 1
    c = np.polyfit(p[:, 1], p[:, 0], deg)[::-1]
    est.coeffs = np.r_[c, np.zeros(3 - len(c))]
    est.center_pts = p
    est.y_min, est.y_max = float(p[:, 1].min()), float(p[:, 1].max())
    est.valid = True
    return est


def simulate(prof, seconds=30.0, seed=0):
    rng = np.random.default_rng(seed)
    cfg = prof.control
    ctrl = Controller(copy.deepcopy(cfg))
    track = Oval()
    L = cfg.wheelbase_m
    dmax = math.radians(cfg.max_steer_deg)
    pos, yaw = track.point(0.0)[0].copy(), 0.0
    s_car, v, delta = 0.0, 1.0, 0.0
    cmd_queue = [(0.0, 0.0)] * max(1, int(round(DELAY_S / DT)))
    lats, steers, wide, laps, lap_start, dist, vs, thrs = [], [], 0, [], 0.0, 0.0, [], []
    t = 0.0
    while t < seconds:
        est = observe(track, pos, yaw, s_car, rng)
        cmd = ctrl.update(est, t)
        cmd_queue.append((cmd.steering, cmd.throttle))
        steer, thr = cmd_queue.pop(0)
        # actuators
        delta += (steer * dmax - delta) * min(1.0, DT / SERVO_TAU)
        v_target = max(0.0, V_GAIN * (thr - V_OFFSET))
        v += (v_target - v) * min(1.0, DT / V_TAU)
        # kinematics with a lateral grip limit (understeer)
        kappa = math.tan(delta) / L / (1 + UNDERSTEER_K * v * v)
        k_max = A_LAT_MAX / max(v * v, 1e-6)
        kappa = max(-k_max, min(k_max, kappa))
        yaw += -v * kappa * DT        # +delta = right turn = yaw decreasing (CCW frame)
        pos += v * DT * np.array([math.cos(yaw), math.sin(yaw)])
        s_new, lat = track.project(pos, s_car)
        ds = (s_new - s_car + track.length / 2) % track.length - track.length / 2
        dist += ds
        s_car = s_new
        if dist - lap_start >= track.length:
            laps.append(t)                       # time at each completed lap
            lap_start += track.length
        lats.append(lat)
        steers.append(steer)
        vs.append(v)
        thrs.append(thr)
        if abs(lat) > 0.30:           # beyond lane edge (0.2) + half car: off track
            wide += 1
        if abs(lat) > 0.6:
            return dict(crashed=True, t=t, laps=[], lat_rms=float("nan"))
        t += DT
    lap_times = np.diff(laps) if len(laps) > 1 else np.array([])     # skip the standing-start lap
    lats = np.array(lats[int(5 / DT):])
    st = np.array(steers[int(5 / DT):])
    return dict(crashed=False, laps=lap_times.tolist(),
                lap=float(np.mean(lap_times)) if len(lap_times) else float("nan"),
                lat_rms=float(np.sqrt(np.mean(lats ** 2))), lat_p95=float(np.percentile(np.abs(lats), 95)),
                lat_mean=float(np.mean(lats)), off=float(wide / len(steers)),
                wobble=float(np.std(np.diff(st)) * 60), sat=float(np.mean(np.abs(st) > 0.98)),
                v=float(np.mean(vs)), thr=float(np.mean(thrs[int(5 / DT):])))


def fmt(r):
    if r["crashed"]:
        return f"CRASHED at {r['t']:.1f}s"
    return (f"lap {r['lap']:.2f}s  lateral rms {r['lat_rms'] * 100:4.1f}cm p95 {r['lat_p95'] * 100:4.1f}cm "
            f"mean {r['lat_mean'] * 100:+5.1f}cm (+ = outside)  off-track {r['off'] * 100:4.1f}%  "
            f"steer-rate {r['wobble']:.2f}/s  sat {r['sat'] * 100:3.0f}%  thr {r['thr']:.3f} v {r['v']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--understeer", type=float, help="override UNDERSTEER_K")
    ap.add_argument("--grip", type=float, help="override A_LAT_MAX")
    ap.add_argument("--delay", type=float, help="override DELAY_S")
    args = ap.parse_args()
    global UNDERSTEER_K, A_LAT_MAX, DELAY_S
    if args.delay is not None:
        DELAY_S = args.delay
    if args.understeer is not None:
        UNDERSTEER_K = args.understeer
    if args.grip is not None:
        A_LAT_MAX = args.grip
    base = yaml.safe_load(open(args.profile))
    for ov in args.set:
        profile.apply_override(base, ov)
    keys = [g.split("=")[0] for g in args.grid]
    vals = [g.split("=")[1].split(",") for g in args.grid]
    for combo in itertools.product(*vals) if keys else [()]:
        d = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            profile.apply_override(d, f"{k}={v}")
        r = simulate(profile.from_dict(d), args.seconds)
        label = " ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo)) or "profile"
        print(f"{label:38s} {fmt(r)}")


if __name__ == "__main__":
    main()
