"""Session profile: every tuning value lives here, loaded from profiles/*.yaml.

Unknown keys raise an error so a typo in a profile cannot silently fall back
to a default.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class CameraCfg:
    width: int = 320
    height: int = 240
    sensor_width: int = 640          # OV5647 640x480 mode = full FoV, ~58 fps
    sensor_height: int = 480
    frame_rate: float = 60.0
    hflip: bool = True               # camera is mounted upside down
    vflip: bool = True
    # Locked values written by tools/calibrate.py. null = leave auto on.
    exposure_us: Optional[int] = None
    analogue_gain: Optional[float] = None
    colour_gains: Optional[List[float]] = None  # [red, blue]


@dataclass
class LensCfg:
    """Pinhole + radial distortion, in units of the *processing* image size.

    fx/fy/cx/cy are fractions of image width/height so the same values work at
    any resolution. Refine with tools/lens_cal.py (chessboard)."""
    fx: float = 0.42
    fy: float = 0.56
    cx: float = 0.5
    cy: float = 0.5
    k1: float = -0.30
    k2: float = 0.08


@dataclass
class GeometryCfg:
    """Camera pose, used to build the image->ground homography when
    `ground_homography` is null. Ground frame: origin at rear axle,
    x = right (m), y = forward (m)."""
    height_m: float = 0.12
    pitch_deg: float = 12.0          # downward tilt
    x_offset_m: float = 0.20         # camera ahead of rear axle


@dataclass
class VisionCfg:
    mode: str = "two_lines"          # two_lines | center_line
    hsv_lower: List[int] = field(default_factory=lambda: [0, 90, 90])     # orange tape
    hsv_upper: List[int] = field(default_factory=lambda: [22, 255, 255])
    roi_top: float = 0.40            # fraction of height; rows above are ignored
    roi_bottom: float = 0.95         # bumper posts are black, the colour threshold skips them
    row_step: int = 2                # scan every Nth row
    blur: int = 3                    # median blur kernel (0/1 = off)
    min_cluster_px: int = 2          # min run of tape pixels in a row
    max_range_m: float = 1.8         # ignore points farther than this
    max_gap_m: float = 0.30          # stop scanning after a forward gap this long
    outlier_m: float = 0.06          # centre points this far off the first fit are dropped
    track_width_m: float = 0.60      # distance between the two tape lines
    track_direction: str = "both"    # left | right | both: turn direction of a one-way track;
                                     # with left/right a lone visible line is taken as the outer line
    assoc_tolerance: float = 0.35    # fraction of track width
    min_points: int = 4              # centre points needed for a valid estimate
    lens: LensCfg = field(default_factory=LensCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    # 3x3 matrix mapping *undistorted normalised* image coords -> ground (m).
    # Written by tools/ground_cal.py; overrides `geometry` when set.
    ground_homography: Optional[List[List[float]]] = None


@dataclass
class ActuatorCfg:
    i2c_bus: int = 1
    pca9685_addr: int = 0x40
    pwm_freq_hz: float = 60.0
    steering_channel: int = 0
    throttle_channel: int = 1
    # Pulse widths in microseconds (from ~/mycar/myconfig.py DonkeyCar calibration).
    steering_left_us: int = 2035
    steering_center_us: int = 1689
    steering_right_us: int = 1343
    throttle_forward_us: int = 2035
    throttle_stop_us: int = 1506
    throttle_reverse_us: int = 1302
    throttle_max: float = 0.35       # hard cap on |forward throttle|
    reverse_max: float = 0.5         # hard cap on |reverse/brake throttle|
    ina219_addr: int = 0x42          # battery monitor (2S pack, 6.0-8.4 V)


@dataclass
class ControlCfg:
    wheelbase_m: float = 0.26
    max_steer_deg: float = 25.0
    lookahead_m: float = 0.60
    steer_smoothing: float = 0.0     # EMA factor 0..1 (0 = none)
    straight_throttle: float = 0.25
    turn_throttle: float = 0.20
    brake_preview_m: float = 1.2     # look this far ahead for an upcoming turn (braking point)
    turn_kappa_enter: float = 0.4    # 1/m; arc curvature to the preview point that counts as a turn
    turn_kappa_exit: float = 0.25
    turn_exit_frames: int = 3
    brake_throttle: float = 0.0      # negative = brake pulse on turn entry
    brake_time_s: float = 0.0
    racing_line_offset_m: float = 0.0  # shift target toward the inside, scaled by turn factor
    speed_blend: bool = True         # throttle blends straight->turn with the turn factor
    turn_factor_smoothing: float = 0.5  # EMA on the falling turn factor (0 = none)
    lookahead_speed_gain: float = 0.0   # extra lookahead (m) per unit throttle above ref
    lookahead_ref_throttle: float = 0.36
    lost_hold_s: float = 0.25        # hold last steering this long when track is lost


@dataclass
class SafetyCfg:
    deadman_button: str = "rb"
    estop_button: str = "b"
    mode_button: str = "x"
    lap_button: str = "a"
    frame_timeout_s: float = 0.10
    deadman_grace_s: float = 0.20    # ignore RB releases shorter than this (F710 wireless dropouts)
    max_run_s: float = 0.0           # 0 = no limit; auto mode stops after this
    battery_min_v: float = 7.0
    manual_throttle_scale: float = 0.3
    joystick_device: str = "/dev/input/js0"


@dataclass
class Profile:
    name: str = "default"
    camera: CameraCfg = field(default_factory=CameraCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    actuators: ActuatorCfg = field(default_factory=ActuatorCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)


def _build(cls, data: dict, path: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        default = getattr(cls(), k)
        if dataclasses.is_dataclass(default):
            kwargs[k] = _build(type(default), v, f"{path}.{k}")
        else:
            kwargs[k] = v
    return cls(**kwargs)


def from_dict(data: dict) -> Profile:
    return _build(Profile, data or {}, "profile")


def to_dict(p: Profile) -> dict:
    return dataclasses.asdict(p)


def load(path: str | Path, overrides: Optional[List[str]] = None) -> Profile:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for ov in overrides or []:
        apply_override(data, ov)
    return from_dict(data)


def save(p: Profile, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(to_dict(p), f, Dumper=_Dumper, sort_keys=False)


class _Dumper(yaml.SafeDumper):
    """Block style for mappings, flow style for short lists of numbers."""


def _repr_list(dumper, data):
    flow = all(not isinstance(v, (dict, list)) for v in data) or \
        all(isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def apply_override(data: dict, spec: str) -> None:
    """Apply 'control.lookahead_m=0.8' to a nested dict (value parsed as YAML)."""
    key, _, raw = spec.partition("=")
    if not _:
        raise ValueError(f"override must be key=value: {spec!r}")
    parts = key.strip().split(".")
    node: Any = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)
