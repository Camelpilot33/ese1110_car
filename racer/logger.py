"""Frames + state to logs/<timestamp>_<mode>/ on a background thread.

Layout:
  frames/000123.jpg   raw camera frame (what vision saw)
  state.jsonl         one JSON object per loop iteration
  profile.yaml        profile used for the run
  events.jsonl        laps, mode changes, e-stops
The control loop never blocks on disk: when the queue is full, frames are
dropped and counted.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import profile as profile_mod


class RunLogger:
    def __init__(self, root: str | Path, tag: str, prof: profile_mod.Profile,
                 save_frames: bool = True, jpeg_quality: int = 85):
        self.dir = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
        (self.dir / "frames").mkdir(parents=True)
        profile_mod.save(prof, self.dir / "profile.yaml")
        self.save_frames = save_frames
        self.quality = jpeg_quality
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._state = open(self.dir / "state.jsonl", "w")
        self._events = open(self.dir / "events.jsonl", "w")
        self._thread = threading.Thread(target=self._run, daemon=True, name="logger")
        self._thread.start()

    def log(self, state: dict, frame: Optional[np.ndarray] = None) -> None:
        try:
            self._q.put_nowait((state, frame if self.save_frames else None))
        except queue.Full:
            self.dropped += 1

    def event(self, kind: str, **data) -> None:
        """Queued like state rows: the control loop never touches the SD card."""
        rec = {"t": time.monotonic(), "event": kind, **data}
        try:
            self._q.put_nowait(("event", rec))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            if item[0] == "event":
                self._events.write(json.dumps(item[1], default=_jsonable) + "\n")
                self._events.flush()
                continue
            state, frame = item
            if frame is not None:
                cv2.imwrite(str(self.dir / "frames" / f"{state['frame']:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self._state.write(json.dumps(state, default=_jsonable) + "\n")

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)
        self._state.close()
        self._events.close()


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
