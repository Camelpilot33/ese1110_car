"""Model-predictive racing controller (control.mode = "mpc").

Everything is planned in the car's own frame (x right, y forward, origin at the
rear axle), where the car's pose is known exactly: at the origin, pointing +y.

Per frame:
  1. LineTracker (state estimation): keeps the last trusted centre line and
     moves it with the car's own motion (speed estimate + steering actually
     acting on the car). A new vision fit is compared with that prediction:
     if it agrees it is blended in, if it jumps (a vision glitch) the
     predicted line is used instead for a few frames. Speed is estimated from
     the throttle history (no encoder or IMU on this car).
  2. Plan (the MPC step, spatial domain): from the car's delay-compensated
     pose, choose lateral offsets d_i from the observed centre-line samples
     that minimise
         curvature^2 of the driven line  +  w_len * its length
     subject to |d_i| <= half width - margin. The line starts exactly at the
     car with its heading, so no pose has to be extrapolated back from the
     camera. Only the first part is executed; the plan is re-solved every
     frame (receding horizon). The curvature term straightens the line early
     on a turn exit; the length term pulls it inside on this near-circular
     oval (minimum curvature alone rides the outside line).
  3. Speed: along the planned line, v <= sqrt(a_lat / |kappa|), with braking and
     acceleration limits and a safe speed at the end of what the camera sees.
     The car accelerates as soon as the planned line opens up.
  4. Steering: pure pursuit to the planned line track_lookahead_m ahead,
     converted to a steering angle with an understeer correction.

All distances are in the profile's ground units (camera geometry fit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .profile import MpcCfg
from .vision import TrackEstimate


# ----------------------------------------------------------------------------- geometry

def ego_transform(points: np.ndarray, ds: float, kappa: float) -> np.ndarray:
    """Express car-frame points in the frame the car has after driving ds along
    an arc of curvature kappa (+ = right)."""
    if len(points) == 0 or ds == 0:
        return points
    th = kappa * ds
    tx, ty = ds * math.sin(th / 2), ds * math.cos(th / 2)
    c, s = math.cos(th), math.sin(th)
    x, y = points[:, 0] - tx, points[:, 1] - ty
    return np.c_[c * x - s * y, s * x + c * y]


def discrete_curvature(p: np.ndarray) -> np.ndarray:
    """Signed curvature (+ = right) at each interior point of a polyline,
    endpoints copied from neighbours."""
    if len(p) < 3:
        return np.zeros(len(p))
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]          # + = turning left
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    k = -2 * cross / np.maximum(den, 1e-9)                       # + = right
    return np.r_[k[0], k, k[-1]]


def resample(poly: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """n points at equal arc length along a polyline, unit right normals, spacing."""
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    si = np.linspace(0.0, s[-1], n)
    pts = np.c_[np.interp(si, s, poly[:, 0]), np.interp(si, s, poly[:, 1])]
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[tang[:, 1], -tang[:, 0]]                         # right of (tx, ty)
    return pts, nrm, s[-1] / max(n - 1, 1)


# ----------------------------------------------------------------------------- estimation

class LineTracker:
    """Centre line in the car frame, carried with ego-motion and gated against
    vision. The car's own pose needs no estimate: it is the frame's origin."""

    def __init__(self, cfg: MpcCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.line: Optional[np.ndarray] = None      # polyline, ordered by distance
        self.coasted = 0
        self.v = 0.0

    def predict(self, dt: float, kappa_car: float, throttle: float) -> None:
        cfg = self.cfg
        if dt <= 0:
            return
        v_ss = max(0.0, cfg.speed_gain * (throttle - cfg.speed_offset))
        self.v += (v_ss - self.v) * min(1.0, dt / cfg.speed_tau_s)
        if self.line is not None:
            self.line = ego_transform(self.line, self.v * dt, kappa_car)
            self.line = self.line[self.line[:, 1] > -0.05]          # drop what is behind
            if len(self.line) < 3:
                self.line = None

    @staticmethod
    def measured_line(est: TrackEstimate, n: int = 15) -> np.ndarray:
        """Centre line points over the observed range only (no extrapolation)."""
        ys = np.linspace(est.y_min, est.y_max, n)
        c = est.coeffs
        return np.c_[c[0] + c[1] * ys + c[2] * ys * ys, ys]

    def update(self, est: Optional[TrackEstimate]) -> Tuple[Optional[np.ndarray], bool]:
        """Returns (line to plan on, gated?)."""
        cfg = self.cfg
        if est is None or not est.valid:
            self.coasted += 1
            return (self.line if self.coasted <= cfg.max_coast else None), True
        meas = self.measured_line(est)
        if self.line is None:
            self.line, self.coasted = meas, 0
            return self.line, False
        pred = self.line
        lo, hi = max(meas[0, 1], pred[0, 1]), min(meas[-1, 1], pred[-1, 1])
        if hi - lo > 0.05:
            ys = np.linspace(lo, hi, 8)
            dx = np.interp(ys, meas[:, 1], meas[:, 0]) - np.interp(ys, pred[:, 1], pred[:, 0])
            innov = float(np.median(np.abs(dx)))
        else:
            innov = 0.0                                  # no overlap: nothing to check against
        if innov > cfg.gate_m:
            if self.coasted < cfg.max_coast:
                self.coasted += 1                        # vision glitch: keep the prediction
                return self.line, True
            self.line, self.coasted = meas, 0            # vision keeps disagreeing: re-initialise
            return self.line, False
        # accept: blend the measurement with the prediction where they overlap
        a = cfg.line_blend
        x_pred = np.interp(meas[:, 1], pred[:, 1], pred[:, 0], left=np.nan, right=np.nan)
        x_new = np.where(np.isnan(x_pred), meas[:, 0], a * meas[:, 0] + (1 - a) * np.nan_to_num(x_pred))
        self.line = np.c_[x_new, meas[:, 1]]
        self.coasted = 0
        return self.line, False


# ----------------------------------------------------------------------------- planning

@dataclass
class Plan:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # incl. the car at [0]
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))       # d_i at the samples
    kappa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ds: float = 0.0


def plan_line(centre: np.ndarray, cfg: MpcCfg, half_width: float, kappa_prev: float,
              start: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Optional[Plan]:
    """Racing line from the car's pose `start` = (x, y, heading; + heading =
    right of +y) through the lane around the observed centre polyline.

    Points: P_-1 (virtual, fixes the heading), P_0 = car, P_i = C_i + d_i N_i.
    Minimise sum |p''|^2 + w_rate (kappa_start - kappa_prev)^2 + w_centre |d|^2
             + w_len * sum(-kappa_c,i * d_i * ds)
    subject to |d_i| <= half_width - margin. Box-constrained QP, active set."""
    n = cfg.horizon_n
    if centre is None or len(centre) < 3 or centre[-1, 1] - centre[0, 1] < 0.15:
        return None
    C, Nrm, ds = resample(centre, n)
    x0, y0, h0 = start
    hdg = np.array([math.sin(h0), math.cos(h0)])
    P0 = np.array([x0, y0])
    h_first = max(float(np.linalg.norm(C[0] - P0)), 1e-3)
    Pm1 = P0 - hdg * h_first                     # virtual point behind: fixes the start heading
    seq = [("fix", Pm1), ("fix", P0)] + [("var", j) for j in range(n)]
    pos = [Pm1, P0] + [C[j] for j in range(n)]

    rows, rhs = [], []
    for k in range(1, len(seq) - 1):
        h1 = max(float(np.linalg.norm(pos[k] - pos[k - 1])), 1e-3)
        h2 = max(float(np.linalg.norm(pos[k + 1] - pos[k])), 1e-3)
        coef = (2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2)))   # p'' weights
        w = cfg.w_first if k == 1 else 1.0
        for axis in (0, 1):
            r = np.zeros(n)
            const = 0.0
            for (kind, val), cw in zip(seq[k - 1:k + 2], coef):
                if kind == "fix":
                    const += cw * val[axis]
                else:
                    r[val] += cw * Nrm[val, axis]
                    const += cw * C[val, axis]
            rows.append(w * r)
            rhs.append(-w * const)
            if k == 1 and axis == 0:
                # steering rate: lateral p'' at the car ~ start curvature
                rows.append(cfg.w_rate * r)
                rhs.append(cfg.w_rate * (kappa_prev - const))
    A = np.vstack(rows + [cfg.w_centre * np.eye(n)])
    b = np.r_[np.array(rhs), np.zeros(n)]
    kc = discrete_curvature(C)
    kc = np.where(np.abs(kc) < cfg.kc_deadband, 0.0, kc)      # fit noise on straights is not a bend
    g = cfg.w_len * (-kc) * ds

    H = 2 * A.T @ A
    f = -2 * A.T @ b + g
    bound = max(0.0, half_width - cfg.margin_m)
    fixed = {}
    d = np.zeros(n)
    for _ in range(10):
        free = [j for j in range(n) if j not in fixed]
        fx = list(fixed)
        d_fix = np.array([fixed[j] for j in fx]) if fx else np.zeros(0)
        rhs_f = -f[free] - (H[np.ix_(free, fx)] @ d_fix if fx else 0.0)
        sol = np.linalg.solve(H[np.ix_(free, free)], rhs_f)
        d[:] = 0.0
        if fx:
            d[fx] = d_fix
        d[free] = sol
        viol = [j for j in free if abs(d[j]) > bound + 1e-6]
        if not viol:
            break
        for j in viol:
            fixed[j] = math.copysign(bound, d[j])
    P = np.vstack([P0, C + d[:, None] * Nrm])
    k = discrete_curvature(np.vstack([Pm1, P]))[1:]
    return Plan(points=P, offsets=d, kappa=k, ds=ds)


def speed_profile(plan: Plan, v_now: float, cfg: MpcCfg) -> np.ndarray:
    # The first segments bend to correct the car's current pose; that is not
    # the corner. Use the line ahead, lightly smoothed, for the speed limit.
    k = np.abs(plan.kappa).copy()
    k[:cfg.speed_skip_n] = k[min(cfg.speed_skip_n, len(k) - 1)]
    k = np.convolve(np.r_[k[0], k, k[-1]], np.ones(3) / 3, "valid")
    v = np.minimum(np.sqrt(cfg.a_lat / np.maximum(k, 1e-3)), cfg.v_max)
    v[-1] = min(v[-1], math.sqrt(cfg.a_lat * cfg.r_unseen_m))    # a turn may start just past the view
    seg = np.hypot(*np.diff(plan.points, axis=0).T)
    for i in range(len(v) - 2, -1, -1):                          # braking
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * cfg.a_brake * seg[i]))
    v[0] = min(v[0], max(v_now, 0.0))                            # start from the current speed
    for i in range(1, len(v)):                                   # acceleration
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * cfg.a_accel * seg[i - 1]))
    return v


def point_at_distance(points: np.ndarray, ld: float) -> Tuple[float, float]:
    """First point along a polyline (car frame) at straight-line distance ld
    from the rear axle; the last point if the line is shorter."""
    d = np.hypot(points[:, 0], points[:, 1])
    idx = np.flatnonzero(d >= ld)
    if len(idx) == 0:
        return float(points[-1, 0]), float(points[-1, 1])
    i = max(1, int(idx[0]))
    a = (ld - d[i - 1]) / max(d[i] - d[i - 1], 1e-9)
    p = points[i - 1] + a * (points[i] - points[i - 1])
    return float(p[0]), float(p[1])


# ----------------------------------------------------------------------------- controller

@dataclass
class MpcOutput:
    steering: float
    throttle: float
    plan: Optional[Plan]
    v_est: float
    v_target: float
    gated: bool
    coasted: int


class MpcController:
    def __init__(self, cfg: MpcCfg, wheelbase: float, max_steer_deg: float,
                 steer_min: float, steer_max: float, half_width: float):
        self.cfg = cfg
        self.L = wheelbase
        self.dmax = math.radians(max_steer_deg)
        self.steer_min, self.steer_max = steer_min, steer_max
        self.half_width = half_width
        self.tracker = LineTracker(cfg)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._last_t: Optional[float] = None
        self._kappa_cmd = 0.0
        self._steer = 0.0
        self._throttle = 0.0
        self._target: Optional[Tuple[float, float]] = None
        self._cmd_hist: List[Tuple[float, float]] = []

    def set_applied(self, steering: float, throttle: float) -> None:
        """Replay only: tell the estimator what was actually sent to the car."""
        self._steer, self._throttle = steering, throttle

    def delayed_steer(self, now: float) -> float:
        self._cmd_hist.append((now, self._steer))
        t_eff = now - self.cfg.steer_delay_s
        while len(self._cmd_hist) > 1 and self._cmd_hist[1][0] <= t_eff:
            self._cmd_hist.pop(0)
        return self._cmd_hist[0][1]

    # steering command <-> path curvature (the car turns less at speed)
    def kappa_from_steer(self, steer: float, v: float) -> float:
        return math.tan(steer * self.dmax) / self.L / (1 + self.cfg.understeer_k * v * v)

    def steer_from_kappa(self, kappa: float, v: float) -> float:
        delta = math.atan(self.L * kappa * (1 + self.cfg.understeer_k * v * v))
        return float(np.clip(delta / self.dmax, self.steer_min, self.steer_max))

    @property
    def v(self) -> float:
        return self.tracker.v

    def update(self, est: TrackEstimate, now: float) -> MpcOutput:
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now
        k_act = self.kappa_from_steer(self.delayed_steer(now), self.tracker.v)
        self.tracker.predict(dt, k_act, self._throttle)
        centre, gated = self.tracker.update(est)
        v = self.tracker.v
        safe_thr = cfg.thr_min if v < 0.3 else cfg.thr_brake
        if centre is None:
            return MpcOutput(self._steer, safe_thr, None, v, 0.0, gated, self.tracker.coasted)
        # delay compensation: plan from where the car will be when the command acts
        ds_lat = v * cfg.latency_s
        fut = ego_transform(np.zeros((1, 2)), -ds_lat, k_act)[0]       # future position, current frame
        plan = plan_line(centre, cfg, self.half_width, self._kappa_cmd,
                         (float(fut[0]), float(fut[1]), k_act * ds_lat))
        if plan is None:
            return MpcOutput(self._steer, safe_thr, None, v, 0.0, gated, self.tracker.coasted)

        # steering: pure pursuit to the planned line track_lookahead_m ahead
        ld = cfg.track_lookahead_m + cfg.track_lookahead_per_v * v
        tx, ty = point_at_distance(plan.points, ld)
        k_cmd = 2.0 * tx / max(tx * tx + ty * ty, 1e-6)
        self._kappa_cmd = k_cmd
        self._steer = self.steer_from_kappa(k_cmd, v)
        self._target = (tx, ty)

        # speed: profile along the plan, read a little ahead (actuation lead)
        prof = speed_profile(plan, v, cfg)
        plan.speed = prof
        dist = np.r_[0.0, np.cumsum(np.hypot(*np.diff(plan.points, axis=0).T))]
        v_target = float(np.interp(max(v, 0.3) * cfg.speed_lead_s, dist, prof))
        thr = cfg.speed_offset + v_target / cfg.speed_gain
        if v_target < v - cfg.brake_margin:
            thr = cfg.thr_brake                                    # coast / brake
        thr = float(np.clip(thr, min(cfg.thr_min, cfg.thr_brake), cfg.thr_max))
        if thr < cfg.thr_min and v < 0.3:
            thr = cfg.thr_min                                       # never stall from rest
        self._throttle = thr
        return MpcOutput(self._steer, thr, plan, v, v_target, gated, self.tracker.coasted)
