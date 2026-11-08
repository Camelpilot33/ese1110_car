"""Image <-> ground-plane mapping.

Pipeline for a pixel (u, v) in the processing image:
  1. undistort with the lens model  -> normalised camera coords (xn, yn)
  2. homography H_n2g               -> ground (x right, y forward) in metres
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .profile import GeometryCfg, LensCfg


def camera_matrix(lens: LensCfg, width: int, height: int) -> np.ndarray:
    return np.array([[lens.fx * width, 0, lens.cx * width],
                     [0, lens.fy * height, lens.cy * height],
                     [0, 0, 1]], dtype=np.float64)


def dist_coeffs(lens: LensCfg) -> np.ndarray:
    return np.array([lens.k1, lens.k2, 0.0, 0.0], dtype=np.float64)


def ground_to_normalized(geo: GeometryCfg) -> np.ndarray:
    """Homography: ground (X, Y, 1) -> normalised camera coords (homogeneous).

    Camera at (0, x_offset, height), pitched down by pitch_deg.
    Camera axes: x right, y down, z forward.
    """
    t = math.radians(geo.pitch_deg)
    s, c = math.sin(t), math.cos(t)
    h, d = geo.height_m, geo.x_offset_m
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, -s, d * s + h * c],
                     [0.0, c, -d * c + h * s]])


class GroundMapper:
    def __init__(self, lens: LensCfg, geo: GeometryCfg, width: int, height: int,
                 homography: Optional[list] = None):
        self.width, self.height = width, height
        self.K = camera_matrix(lens, width, height)
        self.D = dist_coeffs(lens)
        if homography is not None:
            self.H_n2g = np.asarray(homography, dtype=np.float64)
        else:
            self.H_n2g = np.linalg.inv(ground_to_normalized(geo))
        self.H_g2n = np.linalg.inv(self.H_n2g)

    def pixels_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.D).reshape(-1, 2)

    def pixels_to_ground(self, uv: np.ndarray) -> np.ndarray:
        """Returns Nx2 ground points; rows above the horizon come back as NaN."""
        if len(uv) == 0:
            return np.zeros((0, 2))
        n = self.pixels_to_normalized(uv)
        hom = np.hstack([n, np.ones((len(n), 1))]) @ self.H_n2g.T
        w = hom[:, 2:3]
        out = hom[:, :2] / np.where(np.abs(w) < 1e-9, np.nan, w)
        # Points behind the camera / above the horizon map to negative w.
        out[(w[:, 0] <= 0)] = np.nan
        return out

    def ground_to_pixels(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if len(xy) == 0:
            return np.zeros((0, 2))
        hom = np.hstack([xy, np.ones((len(xy), 1))]) @ self.H_g2n.T
        z = hom[:, 2:3]
        obj = np.hstack([hom[:, :2] / z, np.ones((len(xy), 1))]).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.D)
        uv = uv.reshape(-1, 2)
        uv[z[:, 0] <= 0] = np.nan
        return uv


def fit_homography(pixels: np.ndarray, ground: np.ndarray, lens: LensCfg,
                   width: int, height: int) -> np.ndarray:
    """Fit normalised->ground homography from >=4 measured point pairs."""
    mapper = GroundMapper(lens, GeometryCfg(), width, height)
    n = mapper.pixels_to_normalized(pixels)
    H, _ = cv2.findHomography(n.astype(np.float64),
                              np.asarray(ground, dtype=np.float64), 0)
    if H is None:
        raise ValueError("homography fit failed; check the point pairs")
    return H
