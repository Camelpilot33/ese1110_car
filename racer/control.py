"""Steering + speed control. No hardware imports.

Two modes (control.mode):
  pure_pursuit  geometric steering to the lookahead point; throttle blends
                between straight_throttle and turn_throttle
  pid           PID on the heading error to the lookahead point; speed is a
                controller output: speed_max minus slow-downs for an upcoming
                turn, steering effort and tracking error
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .profile import ControlCfg
from .vision import TrackEstimate

STRAIGHT, TURN = "straight", "turn"


@dataclass
class Command:
    steering: float = 0.0        # -1 = full left, +1 = full right
    throttle: float = 0.0        # -1..1 (negative = brake/reverse)
    target: Optional[Tuple[float, float]] = None
    kappa_preview: float = 0.0
    phase: str = STRAIGHT
    lost: bool = False
    turn_factor: float = 0.0
    error: float = 0.0           # heading error to target (rad, + = right)


def path_x(est: TrackEstimate, y: float) -> float:
    """Lateral position of the path at distance y. Beyond the farthest observed
    point the path continues in a straight line along its tangent there: the
    quadratic itself swings wildly when extrapolated on a sharp, short fit."""
    if y <= est.y_max:
        return est.x_at(y)
    c = est.coeffs
    slope = c[1] + 2 * c[2] * est.y_max
    return est.x_at(est.y_max) + slope * (y - est.y_max)


def lookahead_point(est: TrackEstimate, ld: float) -> Tuple[float, float]:
    """First point along the path whose straight-line distance from the rear
    axle reaches ld. Walks the path (robust on sharp curves, where solving
    x(y)^2 + y^2 = ld^2 by iteration collapses to a point beside the car).
    Never goes past the farthest observed point: extending a short, sharply
    curved fit along its end tangent flipped the bend direction (run 10)."""
    y_lo = max(est.y_min, 0.05)
    y_hi = max(est.y_max, y_lo + 1e-3)        # never past the farthest point actually seen
    ys = np.linspace(y_lo, y_hi, 60)
    xs = np.array([path_x(est, y) for y in ys])
    d = np.hypot(xs, ys)
    idx = np.flatnonzero(d >= ld)
    if len(idx) == 0:
        return float(xs[-1]), float(ys[-1])       # path ends short: farthest point
    i = int(idx[0])
    if i == 0:
        return float(xs[0]), float(ys[0])
    # interpolate between samples i-1 and i
    a = (ld - d[i - 1]) / max(d[i] - d[i - 1], 1e-9)
    return float(xs[i - 1] + a * (xs[i] - xs[i - 1])), float(ys[i - 1] + a * (ys[i] - ys[i - 1]))


def pure_pursuit(x: float, y: float, wheelbase: float, max_steer_rad: float) -> float:
    """Normalised steering command for a target point (x right, y forward)."""
    ld2 = x * x + y * y
    if ld2 < 1e-6:
        return 0.0
    kappa = 2.0 * x / ld2
    delta = math.atan(wheelbase * kappa)
    return float(np.clip(delta / max_steer_rad, -1.0, 1.0))


def preview_curvature(est: TrackEstimate, preview_m: float) -> float:
    """Curvature (1/m, + = right) of the arc from the car to the path point
    `preview_m` ahead, i.e. pure pursuit at the preview distance.

    Depends only on that point's position (not on a second derivative of the
    fit), so it is far less noisy than the fitted path's curvature: on the lab
    oval it separates straights (|k| < 0.12) from turns (|k| > 0.56)."""
    x, y = lookahead_point(est, preview_m)
    d2 = x * x + y * y
    return 2.0 * x / d2 if d2 > 1e-6 else 0.0


class PID:
    """PID with derivative-on-measurement (low-pass filtered) and a clamped
    integral (anti-windup)."""

    def __init__(self, kp: float, ki: float, kd: float, i_limit: float, d_filter: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.d_filter = i_limit, d_filter
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_e: Optional[float] = None
        self.d = 0.0

    def update(self, e: float, dt: float) -> float:
        if self.prev_e is not None and dt > 0:
            raw = (e - self.prev_e) / dt
            a = self.d_filter
            self.d = a * self.d + (1 - a) * raw
        self.prev_e = e
        if dt > 0:
            self.integral = float(np.clip(self.integral + e * dt, -self.i_limit, self.i_limit))
        return self.kp * e + self.ki * self.integral + self.kd * self.d


class Controller:
    def __init__(self, cfg: ControlCfg):
        self.cfg = cfg
        self.phase = STRAIGHT
        self._exit_count = 0
        self._brake_until = 0.0
        self._last_valid_t = -1e9
        self._steer = 0.0
        self._f = 0.0
        self._last_t: Optional[float] = None
        self.pid = PID(cfg.kp, cfg.ki, cfg.kd, cfg.i_limit, cfg.d_filter)

    def update(self, est: TrackEstimate, now: float) -> Command:
        cfg = self.cfg
        if not est.valid:
            lost = now - self._last_valid_t > cfg.lost_hold_s
            if lost:
                self.pid.reset()
                self._last_t = None
            return Command(steering=0.0 if lost else self._steer,
                           throttle=0.0 if lost else cfg.turn_throttle,
                           phase=self.phase, lost=lost)
        self._last_valid_t = now

        kappa = preview_curvature(est, cfg.brake_preview_m)
        # One-way track: a bend the wrong way cannot exist, so such a reading
        # is a bad measurement (seen in run 10 at a turn entry, where it made
        # the car think it was on a straight and accelerate off the track).
        # Hold the previous turn state instead of trusting it.
        wrong_way = ((cfg.track_direction == "left" and kappa > 0) or
                     (cfg.track_direction == "right" and kappa < 0))
        if not wrong_way:
            self._update_phase(kappa, now)

        # How much "turn" is ahead: 0 = straight, 1 = full turn. Rises at
        # once (slow down in time), falls with light smoothing (accelerate as
        # the turn opens, without chattering).
        span = max(cfg.turn_kappa_enter - cfg.turn_kappa_exit, 1e-6)
        f_raw = min(1.0, max(0.0, (abs(kappa) - cfg.turn_kappa_exit) / span))
        a_f = cfg.turn_factor_smoothing
        if wrong_way:
            pass                                   # hold self._f
        else:
            self._f = f_raw if f_raw >= self._f else a_f * self._f + (1 - a_f) * f_raw
        f = self._f
        if cfg.track_direction in ("left", "right"):
            # inside of every turn is known: bias/offset signs never flip
            kappa = -abs(kappa) if cfg.track_direction == "left" else abs(kappa)

        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.2)
        self._last_t = now

        if cfg.mode == "pid":
            return self._pid_step(est, kappa, f, dt, now)

        if now < self._brake_until:
            throttle = cfg.brake_throttle
        elif cfg.speed_blend:
            throttle = cfg.straight_throttle + (cfg.turn_throttle - cfg.straight_throttle) * f
        elif self.phase == TURN:
            throttle = cfg.turn_throttle
        else:
            throttle = cfg.straight_throttle

        # Longer lookahead at higher throttle keeps pure pursuit stable.
        ld = cfg.lookahead_m + cfg.lookahead_speed_gain * max(0.0, throttle - cfg.lookahead_ref_throttle)
        tx, ty = self._target(est, ld, kappa, f)
        steer = pure_pursuit(tx, ty, cfg.wheelbase_m, math.radians(cfg.max_steer_deg))
        a = cfg.steer_smoothing
        self._steer = a * self._steer + (1 - a) * steer
        throttle = self._limit_saturation(throttle)
        return Command(self._steer, throttle, (tx, ty), kappa, self.phase, turn_factor=f,
                       error=math.atan2(tx, ty))

    def _target(self, est: TrackEstimate, ld: float, kappa: float, f: float) -> Tuple[float, float]:
        tx, ty = lookahead_point(est, ld)
        if self.cfg.racing_line_offset_m:
            tx += math.copysign(self.cfg.racing_line_offset_m * f, kappa)  # toward the inside
        return tx, ty

    def _limit_saturation(self, throttle: float) -> float:
        """Near full lock the car cannot turn any tighter: shed speed so it
        can hold the line instead of running wide (understeer limiter)."""
        cfg = self.cfg
        over = abs(self._steer) - cfg.sat_threshold
        if over > 0 and throttle > 0 and cfg.sat_throttle_drop:
            throttle = max(0.0, throttle - cfg.sat_throttle_drop * over / max(1e-6, 1 - cfg.sat_threshold))
        return throttle

    def _pid_step(self, est: TrackEstimate, kappa: float, f: float, dt: float, now: float) -> Command:
        cfg = self.cfg
        tx, ty = self._target(est, cfg.lookahead_m, kappa, f)
        e = math.atan2(tx, ty)                       # heading error to target, + = right
        lim = math.radians(cfg.max_error_deg)
        e = float(np.clip(e, -lim, lim))             # one bad frame cannot kick D
        # Extra steering toward the inside, scaled by the turn factor (0 on
        # straights): the car needs more lock in the turns than the PID asks for.
        bias = math.copysign(cfg.turn_steer_bias * f, kappa) if cfg.turn_steer_bias else 0.0
        steer = float(np.clip(self.pid.update(e, dt) + bias, -1.0, 1.0))
        a = cfg.steer_smoothing
        self._steer = a * self._steer + (1 - a) * steer

        # Speed is a controller output: full speed when the road ahead is
        # straight and the car is tracking well; slower for a coming turn
        # (anticipation), for steering effort and for tracking error.
        speed = (cfg.speed_max
                 - cfg.speed_turn_drop * f
                 - cfg.speed_steer_drop * abs(self._steer)
                 - cfg.speed_error_drop * abs(e))
        speed = self._limit_saturation(speed)
        # Never below speed_min: the car stalls under ~0.35 and would stop mid-turn.
        speed = float(np.clip(speed, cfg.speed_min, cfg.speed_max))
        if now < self._brake_until:
            speed = cfg.brake_throttle
        return Command(self._steer, speed, (tx, ty), kappa, self.phase, turn_factor=f, error=e)

    def _update_phase(self, kappa: float, now: float) -> None:
        cfg = self.cfg
        if self.phase == STRAIGHT and abs(kappa) > cfg.turn_kappa_enter:
            self.phase = TURN
            self._exit_count = 0
            if cfg.brake_time_s > 0:
                self._brake_until = now + cfg.brake_time_s
        elif self.phase == TURN:
            self._exit_count = self._exit_count + 1 if abs(kappa) < cfg.turn_kappa_exit else 0
            if self._exit_count >= cfg.turn_exit_frames:
                self.phase = STRAIGHT
