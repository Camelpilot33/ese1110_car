"""Gamepad (Logitech F710 on /dev/input/js0) + dead-man / watchdog gate.

The motor is allowed to run only when ALL of these hold:
  * dead-man button held
  * gamepad connected
  * e-stop not latched (B latches, START clears)
  * newest camera frame younger than frame_timeout_s
  * optional max_run_s not exceeded in auto mode
"""
from __future__ import annotations

import array
import fcntl
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .profile import SafetyCfg

# Linux evdev codes as reported through the js API (F710 in XInput "X" mode).
BUTTON_NAMES = {0x130: "a", 0x131: "b", 0x133: "x", 0x134: "y", 0x136: "lb",
                0x137: "rb", 0x13a: "back", 0x13b: "start", 0x13c: "mode",
                0x13d: "lstick", 0x13e: "rstick"}
AXIS_NAMES = {0x00: "ljx", 0x01: "ljy", 0x02: "lt", 0x03: "rjx", 0x04: "rjy",
              0x05: "rt", 0x10: "dx", 0x11: "dy"}

# D-pad is reported as hat axes; edges become pseudo-button presses.
DPAD = {"dx": ("dpad_left", "dpad_right"), "dy": ("dpad_up", "dpad_down")}

_JSIOCGAXES, _JSIOCGBUTTONS = 0x80016A11, 0x80016A12
_JSIOCGAXMAP, _JSIOCGBTNMAP = 0x80406A32, 0x80406A34
_JS_EVENT = struct.Struct("IhBB")


class Gamepad:
    """Background reader for the Linux joystick API."""

    def __init__(self, device: str):
        self.device = device
        self.buttons: Dict[str, bool] = {}
        self.axes: Dict[str, float] = {}
        self.connected = False
        self._presses: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gamepad")
        self._thread.start()

    def pressed(self) -> List[str]:
        """Buttons pressed since the last call (edge-triggered)."""
        with self._lock:
            out, self._presses = self._presses, []
        return out

    def held(self, name: str) -> bool:
        return self.connected and self.buttons.get(name, False)

    def axis(self, name: str) -> float:
        return self.axes.get(name, 0.0) if self.connected else 0.0

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with open(self.device, "rb") as js:
                    axis_map, btn_map = self._maps(js)
                    self.connected = True
                    while not self._stop.is_set():
                        ev = js.read(_JS_EVENT.size)
                        if len(ev) < _JS_EVENT.size:
                            break
                        _, value, etype, number = _JS_EVENT.unpack(ev)
                        if etype & 0x01:
                            name = BUTTON_NAMES.get(btn_map[number], hex(btn_map[number]))
                            with self._lock:
                                if value and not (etype & 0x80):
                                    self._presses.append(name)
                                self.buttons[name] = bool(value)
                        elif etype & 0x02:
                            name = AXIS_NAMES.get(axis_map[number], hex(axis_map[number]))
                            v = value / 32767.0
                            if name in DPAD and not (etype & 0x80) and abs(v) > 0.5 \
                                    and abs(self.axes.get(name, 0.0)) <= 0.5:
                                with self._lock:
                                    self._presses.append(DPAD[name][v > 0])
                            self.axes[name] = v
            except OSError:
                pass
            # Disconnected: drop all state so nothing stays "held".
            self.connected = False
            self.buttons.clear()
            self.axes.clear()
            self._stop.wait(0.5)

    @staticmethod
    def _maps(js):
        buf = array.array("B", [0])
        fcntl.ioctl(js, _JSIOCGAXES, buf)
        n_axes = buf[0]
        fcntl.ioctl(js, _JSIOCGBUTTONS, buf)
        n_btn = buf[0]
        am = array.array("B", [0] * 0x40)
        fcntl.ioctl(js, _JSIOCGAXMAP, am)
        bm = array.array("H", [0] * 200)
        fcntl.ioctl(js, _JSIOCGBTNMAP, bm)
        return list(am[:n_axes]), list(bm[:n_btn])


class NullGamepad:
    """For replay/bench: dead-man permanently released unless forced."""
    def __init__(self, held: bool = False):
        self.connected = True
        self._held = held

    def pressed(self) -> List[str]:
        return []

    def held(self, name: str) -> bool:
        return self._held

    def axis(self, name: str) -> float:
        return 0.0

    def close(self) -> None:
        pass


@dataclass
class SafetyGate:
    cfg: SafetyCfg
    estop: bool = False
    run_started: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    _last_held: float = -1e9

    def allowed(self, now: float, deadman: bool, connected: bool,
                frame_age: float, auto: bool) -> bool:
        # Debounce releases: brief dropouts of the button do not stop the car.
        if deadman and connected:
            self._last_held = now
        elif connected and now - self._last_held < self.cfg.deadman_grace_s:
            deadman = True
        r = []
        if not connected:
            r.append("no_gamepad")
        if not deadman:
            r.append("deadman")
        if self.estop:
            r.append("estop")
        if frame_age > self.cfg.frame_timeout_s:
            r.append("stale_frame")
        if auto and deadman:
            if self.run_started is None:
                self.run_started = now
            if self.cfg.max_run_s and now - self.run_started > self.cfg.max_run_s:
                r.append("max_run")
        if not deadman:
            self.run_started = None
        self.reasons = r
        return not r
