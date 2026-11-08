import math

import cv2
import numpy as np
import pytest

from racer import profile
from racer.actuators import PCA9685, MockActuators, steering_pulse_us, throttle_pulse_us
from racer.camera import LatestFrame
from racer.control import Controller, lookahead_point, pure_pursuit
from racer.geometry import GroundMapper, fit_homography
from racer.profile import ActuatorCfg, SafetyCfg
from racer.safety import SafetyGate
from racer.vision import Vision, row_runs

W, H = 320, 240
ORANGE = (30, 110, 230)   # BGR
FLOOR = (150, 140, 135)


def render(prof, lines, width_px=5):
    """Draw ground-frame polylines into a synthetic camera image."""
    m = GroundMapper(prof.vision.lens, prof.vision.geometry, W, H)
    img = np.full((H, W, 3), FLOOR, np.uint8)
    for pts in lines:
        uv = m.ground_to_pixels(np.asarray(pts))
        uv = uv[np.isfinite(uv).all(axis=1)]
        cv2.polylines(img, [uv.astype(np.int32)], False, ORANGE, width_px)
    return img


def straight(x, y0=0.3, y1=2.5):
    ys = np.linspace(y0, y1, 60)
    return np.c_[np.full_like(ys, x), ys]


def arc(x0, radius, y0=0.3, y1=2.0):
    """Arc starting tangent to +y at (x0, 0); radius>0 bends right."""
    ys = np.linspace(y0, y1, 80)
    r = abs(radius)
    ys = ys[ys < r * 0.95]
    dx = r - np.sqrt(r * r - ys * ys)
    return np.c_[x0 + math.copysign(1, radius) * dx, ys]


@pytest.fixture
def prof():
    return profile.from_dict({})


# --- profile -----------------------------------------------------------------

def test_default_profile_file_loads():
    p = profile.load("profiles/default.yaml")
    assert p == profile.Profile()


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="lookahed"):
        profile.from_dict({"control": {"lookahed_m": 1}})


def test_override():
    d = {}
    profile.apply_override(d, "control.lookahead_m=0.8")
    assert profile.from_dict(d).control.lookahead_m == 0.8


def test_roundtrip(tmp_path, prof):
    prof.vision.ground_homography = np.eye(3).tolist()
    profile.save(prof, tmp_path / "p.yaml")
    assert profile.load(tmp_path / "p.yaml") == prof


# --- geometry ----------------------------------------------------------------

def test_ground_pixel_roundtrip(prof):
    m = GroundMapper(prof.vision.lens, prof.vision.geometry, W, H)
    g = np.array([[0.0, 0.5], [-0.3, 1.0], [0.3, 1.5], [0.1, 0.8]])
    back = m.pixels_to_ground(m.ground_to_pixels(g))
    np.testing.assert_allclose(back, g, atol=1e-3)


def test_fit_homography_recovers_geometry(prof):
    m = GroundMapper(prof.vision.lens, prof.vision.geometry, W, H)
    g = np.array([[-0.3, 0.5], [0.3, 0.5], [-0.3, 1.2], [0.3, 1.2], [0.0, 0.8]])
    px = m.ground_to_pixels(g)
    Hm = fit_homography(px, g, prof.vision.lens, W, H)
    m2 = GroundMapper(prof.vision.lens, prof.vision.geometry, W, H, Hm.tolist())
    np.testing.assert_allclose(m2.pixels_to_ground(px), g, atol=1e-4)


# --- vision ------------------------------------------------------------------

def test_row_runs():
    row = np.array([0, 255, 255, 0, 0, 255, 0, 255, 255, 255])
    assert row_runs(row, 2) == [1.5, 8.0]


def test_centered_straight(prof):
    img = render(prof, [straight(-0.3), straight(0.3)])
    est = Vision(prof.vision, W, H).process(img)
    assert est.valid
    assert len(est.left_pts) > 3 and len(est.right_pts) > 3
    assert abs(est.x_at(0.6)) < 0.03
    assert abs(est.circle_curvature()) < 0.1


def test_offset_car_steers_back(prof):
    # Car is 0.1 m right of centre -> track centre at x=-0.1 -> steer left.
    img = render(prof, [straight(-0.4), straight(0.2)])
    est = Vision(prof.vision, W, H).process(img)
    assert est.valid
    assert est.x_at(0.6) == pytest.approx(-0.1, abs=0.04)
    cmd = Controller(prof.control).update(est, 0.0)
    assert cmd.steering < -0.1


def test_single_line_visible(prof):
    img = render(prof, [straight(0.3)])     # only right line
    est = Vision(prof.vision, W, H).process(img)
    assert est.valid and len(est.left_pts) == 0
    assert abs(est.x_at(0.6)) < 0.04


def test_left_turn_detected(prof):
    r = 1.0
    img = render(prof, [arc(-0.3, -(r - 0.3)), arc(0.3, -(r + 0.3))])
    est = Vision(prof.vision, W, H).process(img)
    assert est.valid
    assert est.circle_curvature() < -0.4
    c = Controller(prof.control)
    c.cfg.turn_kappa_enter = 0.4
    cmd = c.update(est, 0.0)
    assert cmd.steering < 0 and cmd.phase == "turn"
    assert cmd.throttle == prof.control.turn_throttle


@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0])
def test_turn_direction_and_ordering(prof, radius):
    v = Vision(prof.vision, W, H)
    kl = v.process(render(prof, [arc(-0.3, -(radius - 0.3)), arc(0.3, -(radius + 0.3))])).circle_curvature()
    v.reset()
    kr = v.process(render(prof, [arc(-0.3, radius + 0.3), arc(0.3, radius - 0.3)])).circle_curvature()
    assert kl < -0.2 and kr > 0.2


def test_tighter_turn_bigger_curvature(prof):
    ks = []
    for r in (3.0, 1.0):
        ks.append(abs(Vision(prof.vision, W, H).process(
            render(prof, [arc(-0.3, -(r - 0.3)), arc(0.3, -(r + 0.3))])).circle_curvature()))
    assert ks[1] > ks[0]


def test_no_tape_invalid(prof):
    img = np.full((H, W, 3), FLOOR, np.uint8)
    assert not Vision(prof.vision, W, H).process(img).valid


def test_vision_is_fast_enough(prof):
    import time
    img = render(prof, [straight(-0.3), straight(0.3)])
    v = Vision(prof.vision, W, H)
    t = time.perf_counter()
    for _ in range(50):
        v.process(img)
    assert (time.perf_counter() - t) / 50 < 0.02   # laptop; Pi 4 is ~3-4x slower


# --- control -----------------------------------------------------------------

def test_pure_pursuit_signs():
    assert pure_pursuit(0.0, 1.0, 0.26, math.radians(25)) == 0
    assert pure_pursuit(0.2, 1.0, 0.26, math.radians(25)) > 0
    assert pure_pursuit(-0.2, 1.0, 0.26, math.radians(25)) < 0
    assert pure_pursuit(1.0, 0.1, 0.26, math.radians(25)) == 1.0


def test_lookahead_distance(prof):
    img = render(prof, [straight(-0.4), straight(0.2)])
    est = Vision(prof.vision, W, H).process(img)
    x, y = lookahead_point(est, 0.6)
    assert math.hypot(x, y) == pytest.approx(0.6, abs=0.02)


def test_lost_track_holds_then_stops(prof):
    c = Controller(prof.control)
    img = render(prof, [straight(-0.4), straight(0.2)])
    est = Vision(prof.vision, W, H).process(img)
    s = c.update(est, 0.0).steering
    from racer.vision import TrackEstimate
    held = c.update(TrackEstimate(), 0.1)
    assert held.steering == s and not held.lost
    gone = c.update(TrackEstimate(), 1.0)
    assert gone.lost and gone.throttle == 0


def test_brake_pulse_on_turn_entry(prof):
    cfg = prof.control
    cfg.brake_throttle, cfg.brake_time_s, cfg.turn_kappa_enter = -0.3, 0.2, 0.4
    img = render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)])
    est = Vision(prof.vision, W, H).process(img)
    c = Controller(cfg)
    assert c.update(est, 0.0).throttle == -0.3
    assert c.update(est, 0.3).throttle == cfg.turn_throttle


# --- actuators ---------------------------------------------------------------

class FakeBus:
    def __init__(self):
        self.regs = {0x00: 0x11}
        self.blocks = {}

    def write_byte_data(self, a, r, v):
        self.regs[r] = v

    def read_byte_data(self, a, r):
        return self.regs.get(r, 0)

    def write_i2c_block_data(self, a, r, data):
        self.blocks[r] = data


def ticks(bus, ch):
    d = bus.blocks[0x06 + 4 * ch]
    return d[2] | d[3] << 8


def test_pca9685_matches_donkey_calibration():
    """Profile microseconds must give the same ticks as ~/mycar/myconfig.py."""
    cfg = ActuatorCfg(throttle_max=1.0, reverse_max=1.0)
    bus = FakeBus()
    pca = PCA9685(bus, 0x40, 60)
    assert bus.regs[0xFE] == 101                # prescale for 60 Hz
    for us, expect in [(cfg.steering_left_us, 500), (cfg.steering_right_us, 330),
                       (cfg.throttle_stop_us, 370), (cfg.throttle_forward_us, 500),
                       (cfg.throttle_reverse_us, 320)]:
        pca.set_pulse_us(0, us)
        assert ticks(bus, 0) == expect


def test_pulse_mapping_and_caps():
    cfg = ActuatorCfg()
    assert steering_pulse_us(cfg, -1) == cfg.steering_left_us
    assert steering_pulse_us(cfg, 1) == cfg.steering_right_us
    assert steering_pulse_us(cfg, 0) == cfg.steering_center_us
    assert throttle_pulse_us(cfg, 0) == cfg.throttle_stop_us
    capped = cfg.throttle_stop_us + cfg.throttle_max * (cfg.throttle_forward_us - cfg.throttle_stop_us)
    assert throttle_pulse_us(cfg, 1.0) == pytest.approx(capped)
    m = MockActuators(cfg)
    m.set(3, 3)
    assert m.steering == 1 and m.throttle == cfg.throttle_max


# --- safety ------------------------------------------------------------------

def test_safety_gate():
    g = SafetyGate(SafetyCfg(max_run_s=5))
    assert g.allowed(0, True, True, 0.02, auto=True)
    assert not g.allowed(1, False, True, 0.02, auto=True) and g.reasons == ["deadman"]
    assert not g.allowed(1, True, True, 0.5, auto=True) and "stale_frame" in g.reasons
    assert not g.allowed(1, True, False, 0.02, auto=False)
    g.estop = True
    assert not g.allowed(1, True, True, 0.02, auto=False)
    g.estop = False
    assert g.allowed(2, True, True, 0.02, auto=True)
    assert not g.allowed(8, True, True, 0.02, auto=True) and "max_run" in g.reasons


def test_latest_frame_newest_only():
    lf = LatestFrame()
    assert lf.get(timeout=0.01) is None
    for i in range(3):
        lf.put(np.zeros(1) + i, float(i))
    img, t, idx = lf.get(-1, timeout=0.01)
    assert idx == 2 and t == 2.0
    assert lf.get(2, timeout=0.01) is None


def test_draw_debug_handles_offscreen_points(prof):
    from racer.vision import TrackEstimate, draw_debug
    v = Vision(prof.vision, W, H)
    est = TrackEstimate(valid=True, coeffs=np.array([0.0, 50.0, 400.0]), y_min=0.1, y_max=3.0,
                        left_pts=np.array([[1e6, 1e6]]))
    out = draw_debug(np.zeros((H, W, 3), np.uint8), v, est, target=(1e5, 1e5))
    assert out.shape == (H, W, 3)
