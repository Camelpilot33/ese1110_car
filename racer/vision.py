"""Tape detection -> centre-line estimate in the ground frame.

Per frame:
  1. HSV threshold inside the ROI (between roi_top and roi_bottom)
  2. scan every `row_step` rows, split tape pixels into runs (clusters)
  3. project run centres to the ground plane (undistort + homography)
  4. walk rows near -> far, assigning runs to the left / right line
  5. centre point = midpoint, or one line +/- track_width/2 when only one is seen
  6. fit x(y) = a + b*y + c*y^2 through the centre points

Runs on the laptop for replay; no Pi-only imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .geometry import GroundMapper
from .profile import VisionCfg


@dataclass
class TrackEstimate:
    valid: bool = False
    coeffs: Optional[np.ndarray] = None       # x(y) = c0 + c1*y + c2*y^2
    y_min: float = 0.0
    y_max: float = 0.0
    center_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    left_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    right_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    mask: Optional[np.ndarray] = None

    def x_at(self, y: float) -> float:
        c = self.coeffs
        return float(c[0] + c[1] * y + c[2] * y * y)

    def curvature_at(self, y: float) -> float:
        """Signed curvature (1/m), positive = path bends right."""
        c = self.coeffs
        d1 = c[1] + 2 * c[2] * y
        return float(2 * c[2] / (1 + d1 * d1) ** 1.5)

    def circle_curvature(self, y_lo: float = 0.0, y_hi: float = 1e9) -> float:
        """Signed curvature (1/m, + = right) from a least-squares circle through
        the centre points with y in [y_lo, y_hi]. Falls back to all points when
        fewer than 5 are in range. More reliable on tight turns than the
        quadratic, which bends badly when extrapolated."""
        pts = self.center_pts
        sel = pts[(pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)]
        if len(sel) < 5:
            sel = pts
        if len(sel) < 3:
            return 0.0
        x, y = sel[:, 0], sel[:, 1]
        A = np.c_[x, y, np.ones_like(x)]
        b = -(x * x + y * y)
        (d, e, f), *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc = -d / 2, -e / 2
        r2 = xc * xc + yc * yc - f
        if r2 <= 0 or r2 > 1e6:
            return 0.0
        return float(np.sign(xc - x.mean()) / np.sqrt(r2))

    def summary(self) -> dict:
        if not self.valid:
            return {"valid": False, "n_left": len(self.left_pts), "n_right": len(self.right_pts)}
        return {"valid": True,
                "coeffs": [round(float(v), 5) for v in self.coeffs],
                "y_range": [round(self.y_min, 3), round(self.y_max, 3)],
                "n_left": len(self.left_pts), "n_right": len(self.right_pts)}


def row_runs(row: np.ndarray, min_len: int) -> List[float]:
    """Centres (pixel x) of runs of nonzero pixels in a 1-D mask row."""
    nz = np.flatnonzero(row)
    if len(nz) == 0:
        return []
    breaks = np.flatnonzero(np.diff(nz) > 1)
    starts = np.r_[nz[0], nz[breaks + 1]]
    ends = np.r_[nz[breaks], nz[-1]]
    keep = (ends - starts + 1) >= min_len
    return list(((starts + ends) / 2.0)[keep])


def mask_runs(sub: np.ndarray, min_len: int):
    """Runs of nonzero pixels in each row of a 2-D mask, vectorised.

    Returns (row index, run centre x) arrays, ordered by row then x."""
    b = (sub > 0).astype(np.int8)
    d = np.diff(np.pad(b, ((0, 0), (1, 1))), axis=1)
    sr, sc = np.nonzero(d == 1)      # run starts (row-major order)
    _, ec = np.nonzero(d == -1)      # run ends (exclusive), same order
    keep = (ec - sc) >= min_len
    return sr[keep], (sc[keep] + ec[keep] - 1) / 2.0


class Vision:
    def __init__(self, cfg: VisionCfg, width: int, height: int):
        self.cfg = cfg
        self.width, self.height = width, height
        self.mapper = GroundMapper(cfg.lens, cfg.geometry, width, height,
                                   cfg.ground_homography)
        self.lower = np.array(cfg.hsv_lower, dtype=np.uint8)
        self.upper = np.array(cfg.hsv_upper, dtype=np.uint8)
        self._prev_center = 0.0   # lateral centre at the nearest row, last frame

    def threshold(self, bgr: np.ndarray) -> np.ndarray:
        h = bgr.shape[0]
        top, bot = int(self.cfg.roi_top * h), int(self.cfg.roi_bottom * h)
        hsv = cv2.cvtColor(bgr[top:bot], cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lower, self.upper)
        if self.cfg.blur and self.cfg.blur > 1:
            m = cv2.medianBlur(m, self.cfg.blur | 1)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask[top:bot] = m
        return mask

    def process(self, bgr: np.ndarray, keep_mask: bool = False) -> TrackEstimate:
        cfg = self.cfg
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        mask = self.threshold(bgr)
        h = mask.shape[0]
        top, bot = int(cfg.roi_top * h), int(cfg.roi_bottom * h)

        # Tape runs on every row_step-th row, found for all rows at once.
        vs = np.arange(bot - 1, top - 1, -cfg.row_step)          # near -> far
        rows, us = mask_runs(mask[vs], cfg.min_cluster_px)
        est = TrackEstimate(mask=mask if keep_mask else None)
        if len(us) == 0:
            return est
        g = self.mapper.pixels_to_ground(np.c_[us, vs[rows]])
        ok = np.isfinite(g).all(axis=1) & (g[:, 1] > 0) & (g[:, 1] < cfg.max_range_m)
        rows, g = rows[ok], g[ok]
        if len(rows) == 0:
            return est
        # rows is sorted (near first); split into per-row groups.
        cuts = np.flatnonzero(np.diff(rows)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, len(rows)]

        w = cfg.track_width_m
        tol = cfg.assoc_tolerance * w
        center = self._prev_center
        left, right, centers = [], [], []
        for a, b in zip(starts.tolist(), ends.tolist()):
            xs = g[a:b, 0]
            y = float(g[a:b, 1].mean())
            if centers and y - centers[-1][1] > cfg.max_gap_m:
                break       # long gap: anything beyond is probably not our track
            if cfg.mode == "center_line":
                i = int(np.argmin(np.abs(xs - center)))
                if abs(xs[i] - center) < tol:
                    center = float(xs[i])
                    centers.append((center, y))
                continue
            lx = self._nearest(xs, center - w / 2, tol)
            rx = self._nearest(xs, center + w / 2, tol)
            if lx is not None and rx is not None and rx - lx < 0.5 * w:
                # Both matched the same blob region; keep the closer one.
                if abs(lx - (center - w / 2)) < abs(rx - (center + w / 2)):
                    rx = None
                else:
                    lx = None
            if lx is not None:
                left.append((lx, y))
            if rx is not None:
                right.append((rx, y))
            if lx is not None and rx is not None:
                cpt = ((lx + rx) / 2, y)
            elif lx is not None:
                cpt = self._offset(left, +w / 2)
            elif rx is not None:
                cpt = self._offset(right, -w / 2)
            else:
                continue
            # Predict where the centre will be on the next (farther) row.
            center = cpt[0] if lx is not None and rx is not None else (
                (lx if lx is not None else rx) + (w / 2 if lx is not None else -w / 2))
            centers.append(cpt)

        # One-way track: a lone line is the OUTER one (for a left-only track the
        # outer line sweeps across the view in every turn; the inner line
        # leaves the field of view to the left).
        if cfg.mode == "two_lines" and cfg.track_direction in ("left", "right"):
            # "Lone" = the other side has only a few stray points.
            def lone(a, b):
                return len(a) >= cfg.min_points and len(b) < max(3, 0.15 * len(a))
            if cfg.track_direction == "left" and lone(left, right):
                right, left = left, []
                centers = self._centers_from_line(right, -w / 2)
            elif cfg.track_direction == "right" and lone(right, left):
                left, right = right, []
                centers = self._centers_from_line(left, +w / 2)

        est.left_pts = np.array(left).reshape(-1, 2)
        est.right_pts = np.array(right).reshape(-1, 2)
        est.center_pts = np.array(centers).reshape(-1, 2)
        if len(centers) < cfg.min_points:
            return est
        cp = est.center_pts
        deg = 2 if len(cp) >= 6 else 1
        coeffs = np.polyfit(cp[:, 1], cp[:, 0], deg)
        # One robust pass: drop points far off the first fit (mis-associations).
        resid = np.abs(np.polyval(coeffs, cp[:, 1]) - cp[:, 0])
        keep = resid < max(cfg.outlier_m, 2.5 * float(np.median(resid)))
        if keep.sum() >= cfg.min_points and not keep.all():
            cp = est.center_pts = cp[keep]
            deg = 2 if len(cp) >= 6 else 1
            coeffs = np.polyfit(cp[:, 1], cp[:, 0], deg)
        coeffs = coeffs[::-1]
        est.coeffs = np.r_[coeffs, np.zeros(3 - len(coeffs))]
        est.y_min, est.y_max = float(cp[:, 1].min()), float(cp[:, 1].max())
        est.valid = True
        self._prev_center = est.x_at(est.y_min)
        return est

    def reset(self) -> None:
        self._prev_center = 0.0

    @classmethod
    def _centers_from_line(cls, line: list, dist: float) -> list:
        return [cls._offset(line[:i + 1], dist) for i in range(len(line))]

    @staticmethod
    def _offset(line: list, dist: float) -> tuple:
        """Point `dist` to the right of the newest point of `line`, measured
        perpendicular to the line's local direction (handles curves)."""
        x, y = line[-1]
        if len(line) >= 2:
            px, py = line[-3] if len(line) >= 3 else line[-2]
            dx, dy = x - px, y - py
            n = math.hypot(dx, dy)
            if n > 1e-6 and dy > 0:
                return x + dist * dy / n, y - dist * dx / n
        return x + dist, y

    @staticmethod
    def _nearest(xs: np.ndarray, target: float, tol: float) -> Optional[float]:
        i = int(np.argmin(np.abs(xs - target)))
        return float(xs[i]) if abs(xs[i] - target) < tol else None


def draw_debug(bgr: np.ndarray, vision: Vision, est: TrackEstimate,
               target: Optional[tuple] = None, text: str = "") -> np.ndarray:
    """Overlay mask, detected lines, fitted centre line and lookahead target."""
    img = bgr.copy()
    if img.shape[1] != vision.width:
        img = cv2.resize(img, (vision.width, vision.height))
    h = img.shape[0]
    cv2.line(img, (0, int(vision.cfg.roi_top * h)), (img.shape[1], int(vision.cfg.roi_top * h)), (80, 80, 80), 1)
    cv2.line(img, (0, int(vision.cfg.roi_bottom * h)), (img.shape[1], int(vision.cfg.roi_bottom * h)), (80, 80, 80), 1)
    if est.mask is not None:
        img[est.mask > 0] = (255, 0, 255)

    def dots(pts, color):
        hh, ww = img.shape[:2]
        for u, v in vision.mapper.ground_to_pixels(pts):
            if np.isfinite(u) and np.isfinite(v) and 0 <= u < ww and 0 <= v < hh:
                cv2.circle(img, (int(u), int(v)), 2, color, -1)

    dots(est.left_pts, (255, 128, 0))
    dots(est.right_pts, (0, 128, 255))
    if est.valid:
        ys = np.linspace(est.y_min, est.y_max, 20)
        dots(np.c_[[est.x_at(y) for y in ys], ys], (0, 255, 0))
    if target is not None:
        uv = vision.mapper.ground_to_pixels(np.array([target]))[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]:
            cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (0, 0, 255), 2)
    if text:
        cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img
