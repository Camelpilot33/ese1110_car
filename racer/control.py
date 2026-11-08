"""Pure pursuit steering + straight/turn speed profile. No hardware imports."""
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


def lookahead_point(est: TrackEstimate, ld: float) -> Tuple[float, float]:
    """Point on the fitted path at straight-line distance ~ld from the rear axle.

    Extrapolates at most 30 % past the farthest observed point."""
    y_cap = max(est.y_max * 1.3, est.y_min)
    y = min(ld, y_cap)
    for _ in range(5):
        x = est.x_at(y)
        y = min(math.sqrt(max(ld * ld - x * x, 1e-6)), y_cap)
    return est.x_at(y), y


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


class Controller:
    def __init__(self, cfg: ControlCfg):
        self.cfg = cfg
        self.phase = STRAIGHT
        self._exit_count = 0
        self._brake_until = 0.0
        self._last_valid_t = -1e9
        self._steer = 0.0

    def update(self, est: TrackEstimate, now: float) -> Command:
        cfg = self.cfg
        if not est.valid:
            lost = now - self._last_valid_t > cfg.lost_hold_s
            return Command(steering=0.0 if lost else self._steer,
                           throttle=0.0 if lost else cfg.turn_throttle,
                           phase=self.phase, lost=lost)
        self._last_valid_t = now

        kappa = preview_curvature(est, cfg.brake_preview_m)
        self._update_phase(kappa, now)

        tx, ty = lookahead_point(est, cfg.lookahead_m)
        if self.phase == TURN and cfg.racing_line_offset_m:
            tx += math.copysign(cfg.racing_line_offset_m, kappa)  # toward the inside
        steer = pure_pursuit(tx, ty, cfg.wheelbase_m, math.radians(cfg.max_steer_deg))
        a = cfg.steer_smoothing
        self._steer = a * self._steer + (1 - a) * steer

        if now < self._brake_until:
            throttle = cfg.brake_throttle
        elif self.phase == TURN:
            throttle = cfg.turn_throttle
        else:
            throttle = cfg.straight_throttle
        return Command(self._steer, throttle, (tx, ty), kappa, self.phase)

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
