"""Threaded frame sources. Consumers always get the newest frame only.

`PiCamera` needs picamera2 (Pi only). `LogSource` replays a logs/<run>/ dir on
any machine.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .profile import CameraCfg

Frame = Tuple[np.ndarray, float, int]   # (BGR image, monotonic time, index)


class LatestFrame:
    """Single-slot mailbox: writer overwrites, reader waits for a newer index."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None

    def put(self, img: np.ndarray, t: float) -> None:
        with self._cond:
            idx = self._frame[2] + 1 if self._frame else 0
            self._frame = (img, t, idx)
            self._cond.notify_all()

    def get(self, newer_than: int = -1, timeout: float = 0.1) -> Optional[Frame]:
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None and self._frame[2] > newer_than,
                                timeout=timeout)
            f = self._frame
            return f if f is not None and f[2] > newer_than else None

    def peek(self) -> Optional[Frame]:
        with self._cond:
            return self._frame


class PiCamera:
    def __init__(self, cfg: CameraCfg, auto: bool = False):
        from libcamera import Transform
        from picamera2 import Picamera2

        self.cfg = cfg
        self.cam = Picamera2()
        controls = {"FrameRate": cfg.frame_rate}
        if not auto:
            controls.update(locked_controls(cfg))
        config = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},  # BGR in memory
            sensor={"output_size": (cfg.sensor_width, cfg.sensor_height)},
            transform=Transform(hflip=cfg.hflip, vflip=cfg.vflip),
            controls=controls, buffer_count=3)
        self.cam.configure(config)
        self.latest = LatestFrame()
        self._stop = threading.Event()
        self.cam.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            img = self.cam.capture_array("main")
            self.latest.put(img, time.monotonic())

    def metadata(self) -> dict:
        return self.cam.capture_metadata()

    def set_controls(self, controls: dict) -> None:
        self.cam.set_controls(controls)

    def get(self, newer_than: int = -1, timeout: float = 0.1) -> Optional[Frame]:
        return self.latest.get(newer_than, timeout)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cam.stop()
        self.cam.close()


def locked_controls(cfg: CameraCfg) -> dict:
    c = {}
    if cfg.exposure_us is not None and cfg.analogue_gain is not None:
        c.update(AeEnable=False, ExposureTime=int(cfg.exposure_us),
                 AnalogueGain=float(cfg.analogue_gain))
    if cfg.colour_gains is not None:
        c.update(AwbEnable=False, ColourGains=tuple(float(g) for g in cfg.colour_gains))
    return c


class LogSource:
    """Iterate frames of a logged run in order (for replay)."""

    def __init__(self, run_dir: str | Path):
        import cv2
        self._cv2 = cv2
        d = Path(run_dir) / "frames"
        self.paths = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        if not self.paths:
            raise FileNotFoundError(f"no frames in {run_dir}/frames")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        for p in self.paths:
            yield int(p.stem), self._cv2.imread(str(p))
