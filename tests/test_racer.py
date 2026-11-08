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
from racer.vision import Vision, mask_runs, row_runs

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
    cfg.brake_throttle, cfg.brake_time_s = -0.3, 0.2
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
    g = SafetyGate(SafetyCfg(max_run_s=5, deadman_grace_s=0))
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


def test_left_only_track_lone_line_is_outer(prof):
    # Car has drifted out: only one line visible, 0.1 m to its LEFT. On a
    # left-only track that is the outer line, so the target must be further left.
    img = render(prof, [straight(-0.1)])
    est = Vision(prof.vision, W, H).process(img)
    assert len(est.left_pts) > 3 and est.x_at(0.6) > 0            # default: taken as left line
    prof.vision.track_direction = "left"
    est = Vision(prof.vision, W, H).process(img)
    assert len(est.right_pts) > 3 and len(est.left_pts) == 0
    assert est.x_at(0.6) == pytest.approx(-0.1 - prof.vision.track_width_m / 2, abs=0.04)
    assert Controller(prof.control).update(est, 0.0).steering < -0.3


def test_left_only_track_keeps_two_lines(prof):
    prof.vision.track_direction = "left"
    est = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    assert len(est.left_pts) > 3 and len(est.right_pts) > 3
    assert abs(est.x_at(0.6)) < 0.03


def test_deadman_grace():
    g = SafetyGate(SafetyCfg(deadman_grace_s=0.2))
    assert g.allowed(0.0, True, True, 0.02, auto=True)
    assert g.allowed(0.1, False, True, 0.02, auto=True)        # brief dropout ignored
    assert not g.allowed(0.25, False, True, 0.02, auto=True)   # real release stops
    assert g.allowed(0.3, True, True, 0.02, auto=True)
    assert not g.allowed(0.35, False, False, 0.02, auto=True)  # disconnect: no grace
    g.estop = True
    assert not g.allowed(0.4, True, True, 0.02, auto=True)     # e-stop is immediate


def test_straight_stays_straight_phase(prof):
    est = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    cmd = Controller(prof.control).update(est, 0.0)
    assert cmd.phase == "straight" and abs(cmd.kappa_preview) < 0.1
    assert cmd.throttle == prof.control.straight_throttle


def test_logger_writes_state_events_frames(tmp_path, prof):
    import json as _json
    from racer.logger import RunLogger
    log = RunLogger(tmp_path, "t", prof)
    log.event("button", button="a")
    log.log({"t": 1.0, "frame": 7, "x": np.float32(0.5)}, np.zeros((8, 8, 3), np.uint8))
    log.close()
    assert (log.dir / "frames" / "000007.png").exists()
    assert _json.loads((log.dir / "state.jsonl").read_text())["x"] == 0.5
    assert _json.loads((log.dir / "events.jsonl").read_text())["button"] == "a"


def test_racing_line_offset_moves_target_inside(prof):
    img = render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)])      # left turn
    est = Vision(prof.vision, W, H).process(img)
    base = Controller(prof.control).update(est, 0.0)
    prof.control.racing_line_offset_m = 0.1
    inside = Controller(prof.control).update(est, 0.0)
    assert base.turn_factor > 0.5
    assert inside.target[0] < base.target[0] - 0.04
    assert inside.steering < base.steering


def test_speed_blend_between_straight_and_turn(prof):
    c = prof.control
    c.straight_throttle, c.turn_throttle = 0.5, 0.3
    est_s = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    est_t = Vision(prof.vision, W, H).process(render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)]))
    ctrl = Controller(c)
    assert ctrl.update(est_s, 0.0).throttle == pytest.approx(0.5)
    turn = ctrl.update(est_t, 0.1).throttle
    assert 0.3 <= turn < 0.4
    # leaving the turn: throttle recovers within a few frames
    th = [ctrl.update(est_s, 0.2 + i * 0.03).throttle for i in range(6)]
    assert th[0] > turn and th[-1] > 0.49


def test_mask_runs_matches_row_runs():
    rng = np.random.default_rng(0)
    m = (rng.random((20, 50)) > 0.6).astype(np.uint8) * 255
    r, u = mask_runs(m, 2)
    for i in range(20):
        assert list(u[r == i]) == row_runs(m[i], 2)


def test_understeer_limiter(prof):
    from racer.vision import TrackEstimate
    est = TrackEstimate(valid=True, coeffs=np.array([-0.5, 0.0, 0.0]), y_min=0.2, y_max=1.0,
                        center_pts=np.c_[np.full(6, -0.5), np.linspace(0.2, 1.0, 6)])
    c = prof.control
    c.straight_throttle = c.turn_throttle = 0.4
    free = Controller(c).update(est, 0.0)
    assert free.steering < -0.98 and free.throttle == pytest.approx(0.4)
    c.sat_threshold, c.sat_throttle_drop = 0.85, 0.06
    assert Controller(c).update(est, 0.0).throttle == pytest.approx(0.34, abs=1e-6)


def _offset_est(x):
    from racer.vision import TrackEstimate
    return TrackEstimate(valid=True, coeffs=np.array([x, 0.0, 0.0]), y_min=0.2, y_max=1.0,
                         center_pts=np.c_[np.full(6, x), np.linspace(0.2, 1.0, 6)])


def test_pid_steers_toward_line_and_integrates(prof):
    c = prof.control
    c.mode, c.kp, c.ki, c.kd = "pid", 1.0, 0.5, 0.0
    ctrl = Controller(c)
    first = ctrl.update(_offset_est(-0.1), 0.0).steering
    assert first < 0                                   # centre to the left -> steer left
    later = [ctrl.update(_offset_est(-0.1), 0.05 * i).steering for i in range(1, 20)]
    assert later[-1] < first                           # integral builds on a steady error


def test_pid_derivative_damps(prof):
    c = prof.control
    c.mode, c.kp, c.ki, c.kd, c.d_filter = "pid", 1.0, 0.0, 0.5, 0.0
    ctrl = Controller(c)
    ctrl.update(_offset_est(-0.2), 0.0)
    s = ctrl.update(_offset_est(-0.1), 0.05).steering   # error shrinking -> D opposes P
    assert s > -math.atan2(0.1, 0.7) * 1.0


def test_pid_speed_output(prof):
    c = prof.control
    c.mode = "pid"
    ctrl = Controller(c)
    straight_cmd = ctrl.update(Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)])), 0.0)
    assert straight_cmd.throttle > c.speed_max - 0.01
    turn_cmd = Controller(c).update(Vision(prof.vision, W, H).process(
        render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)])), 0.0)
    assert c.speed_min <= turn_cmd.throttle < straight_cmd.throttle - 0.05


def test_lookahead_on_sharp_short_fit_stays_ahead(prof):
    """Regression: run 5 turn-entry frames collapsed the target to y=0 (89 deg error)."""
    from racer.vision import TrackEstimate
    for coeffs, y_max in (([0.04, 0.39, -2.07], 0.54), ([0.06, 0.16, -1.83], 0.54),
                          ([-0.25, 2.04, -4.11], 0.49)):
        est = TrackEstimate(valid=True, coeffs=np.array(coeffs), y_min=0.2, y_max=y_max)
        x, y = lookahead_point(est, 0.7)
        assert y > 0.4, (coeffs, x, y)
        assert abs(math.degrees(math.atan2(x, y))) < 50
        assert x < 0          # path bends left -> target left


def test_lone_inner_line_recognised_from_history(prof):
    """Car hugging the inner line: outer line leaves the view. With recent
    history of both lines the lone line must stay the LEFT (inner) one."""
    prof.vision.track_direction = "left"
    v = Vision(prof.vision, W, H)
    w = prof.vision.track_width_m
    est = v.process(render(prof, [straight(-0.12), straight(-0.12 + w)]))
    assert len(est.left_pts) > 3 and len(est.right_pts) > 3
    est = v.process(render(prof, [straight(-0.10)]))          # outer line gone
    assert len(est.left_pts) > 3 and len(est.right_pts) == 0
    assert est.x_at(0.6) > 0                                   # centre to the right of it
    # without history it falls back to the outer-line rule
    v2 = Vision(prof.vision, W, H)
    est2 = v2.process(render(prof, [straight(-0.10)]))
    assert len(est2.right_pts) > 3 and est2.x_at(0.6) < 0


def test_turn_steer_bias_only_in_turns(prof):
    c = prof.control
    c.mode = "pid"
    est_t = Vision(prof.vision, W, H).process(render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)]))
    est_s = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    base_t = Controller(c).update(est_t, 0.0).steering
    base_s = Controller(c).update(est_s, 0.0).steering
    c.turn_steer_bias = 0.15
    assert Controller(c).update(est_t, 0.0).steering == pytest.approx(max(-1.0, base_t - 0.15), abs=1e-6)
    assert Controller(c).update(est_s, 0.0).steering == pytest.approx(base_s, abs=1e-6)


def test_wrong_way_bend_ignored_on_left_track(prof):
    """Run 10: a right-bend reading at a left-turn entry dropped the turn factor
    and the car accelerated off the track. On a left-only track it must hold."""
    from racer.vision import TrackEstimate
    c = prof.control
    c.mode, c.track_direction, c.turn_steer_bias = "pid", "left", 0.15
    ctrl = Controller(c)
    left = TrackEstimate(valid=True, coeffs=np.array([0.0, 0.0, -1.2]), y_min=0.2, y_max=0.75)
    right = TrackEstimate(valid=True, coeffs=np.array([0.0, 0.0, +0.8]), y_min=0.2, y_max=0.75)
    a = ctrl.update(left, 0.0)
    assert a.turn_factor == pytest.approx(1.0)
    b = ctrl.update(right, 0.02)                 # bogus right bend
    assert b.turn_factor == pytest.approx(1.0)   # held
    assert b.throttle < c.speed_max - 0.03       # still turn speed, not straight speed


def test_lookahead_never_extrapolates(prof):
    from racer.vision import TrackEstimate
    est = TrackEstimate(valid=True, coeffs=np.array([0.0, 1.4, -2.7]), y_min=0.25, y_max=0.54)
    x, y = lookahead_point(est, 0.7)
    assert y <= 0.54 + 1e-9


def test_soft_start_ramp(prof):
    c = prof.control
    c.mode, c.start_ramp_s, c.speed_max, c.speed_min = "pid", 1.0, 0.55, 0.38
    est = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    ctrl = Controller(c)
    ctrl.on_start(10.0)
    th = [ctrl.update(est, 10.0 + t).throttle for t in (0.0, 0.5, 1.0, 1.5)]
    assert th[0] == pytest.approx(0.38, abs=1e-6)          # starts at the floor
    assert 0.38 < th[1] < th[2]
    assert th[3] == pytest.approx(ctrl.update(est, 11.6).throttle, abs=1e-3)  # ramp finished
    assert th[3] > 0.5


def test_extended_left_steering_keeps_mapping():
    from racer.actuators import steer_limits
    cfg = ActuatorCfg(steering_left_limit_us=2200)
    lo, hi = steer_limits(cfg)
    assert lo == pytest.approx(-(2200 - 1689) / (2035 - 1689)) and hi == 1.0
    assert steering_pulse_us(cfg, -0.5) == pytest.approx(steering_pulse_us(ActuatorCfg(), -0.5))
    assert steering_pulse_us(cfg, -1.0) == pytest.approx(2035)
    assert steering_pulse_us(cfg, -9.0) == pytest.approx(2200)
    assert steering_pulse_us(cfg, +9.0) == pytest.approx(1343)


def test_controller_uses_extended_left_range(prof):
    from racer.vision import TrackEstimate
    est = TrackEstimate(valid=True, coeffs=np.array([-0.6, 0.0, 0.0]), y_min=0.2, y_max=1.0)
    c = prof.control
    c.mode = "pid"
    assert Controller(c).update(est, 0.0).steering == pytest.approx(-1.0)
    c.steer_min = -1.48
    assert Controller(c).update(est, 0.0).steering < -1.2


# --- mpc ---------------------------------------------------------------------

def _mpc_cfg(prof):
    c = prof.control
    c.mode = "mpc"
    c.mpc.half_width_m = prof.vision.track_width_m / 2
    return c


def _centre(prof, lines):
    from racer.mpc import LineTracker
    est = Vision(prof.vision, W, H).process(render(prof, lines))
    return LineTracker.measured_line(est), est


def test_ego_transform_signs():
    from racer.mpc import ego_transform
    p = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(ego_transform(p, 0.5, 0.0), [[0.0, 0.5]], atol=1e-9)
    # after turning right, a point straight ahead appears to the left
    assert ego_transform(p, 0.3, 1.0)[0, 0] < 0


def test_mpc_plan_straight_centred(prof):
    from racer.mpc import plan_line
    c = _mpc_cfg(prof)
    centre, _ = _centre(prof, [straight(-0.3), straight(0.3)])
    plan = plan_line(centre, c.mpc, 0.3, 0.0)
    assert np.all(np.abs(plan.offsets) < 0.03) and np.all(np.abs(plan.kappa) < 0.3)
    np.testing.assert_allclose(plan.points[0], [0, 0])           # starts at the car


def test_mpc_plan_left_turn_inside_within_lane(prof):
    from racer.mpc import plan_line
    c = _mpc_cfg(prof)
    centre, _ = _centre(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)])
    plan = plan_line(centre, c.mpc, 0.3, 0.0)
    bound = 0.3 - c.mpc.margin_m
    assert np.all(np.abs(plan.offsets) <= bound + 1e-6)
    assert plan.offsets[len(plan.offsets) // 2] < -0.05        # toward the inside (left)
    c.mpc.w_len = 0.0                                          # pure min curvature rides the outside
    assert plan_line(centre, c.mpc, 0.3, 0.0).offsets[-1] > plan.offsets[-1] + 0.05


def test_mpc_plan_starts_with_car_heading(prof):
    """The plan starts at the car with its heading. Inside the allowed band on
    a straight it simply goes straight; outside the band it bends back in."""
    from racer.mpc import plan_line
    c = _mpc_cfg(prof)
    inside, _ = _centre(prof, [straight(-0.45), straight(0.15)])     # car 15 cm right of centre
    plan = plan_line(inside, c.mpc, 0.3, 0.0)
    np.testing.assert_allclose(plan.points[0], [0, 0])
    assert np.all(np.abs(plan.points[:, 0]) < 0.02)                  # no needless swerve
    ys = np.linspace(0.3, 1.2, 15)
    outside = np.c_[np.full_like(ys, -0.35), ys]                     # centre 35 cm left: car outside band
    plan = plan_line(outside, c.mpc, 0.3, 0.0)
    bound = 0.3 - c.mpc.margin_m
    assert np.all(np.abs(plan.offsets) <= bound + 1e-6)
    assert plan.points[-1, 0] < -0.05 and np.min(plan.kappa[:4]) < 0   # bends back left


def test_mpc_rate_term_pulls_start_curvature(prof):
    from racer.mpc import plan_line
    c = _mpc_cfg(prof)
    centre, _ = _centre(prof, [straight(-0.3), straight(0.3)])
    c.mpc.w_rate = 0.5
    a = plan_line(centre, c.mpc, 0.3, +2.0).kappa[0]
    b = plan_line(centre, c.mpc, 0.3, -2.0).kappa[0]
    assert a > b


def test_mpc_controller_turn_vs_straight(prof):
    c = _mpc_cfg(prof)
    est_s = Vision(prof.vision, W, H).process(render(prof, [straight(-0.3), straight(0.3)]))
    est_t = Vision(prof.vision, W, H).process(render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)]))
    outs = {}
    for name, est in (("s", est_s), ("t", est_t)):
        ctrl = Controller(c)
        for i in range(120):
            ctrl.mpc.tracker.line = None          # static picture: no ego-motion carry-over
            outs[name] = ctrl.update(est, i / 60)
    cs, ct = outs["s"], outs["t"]
    assert abs(cs.steering) < 0.15 and ct.steering < -0.3
    assert cs.info["v_target"] > ct.info["v_target"]
    assert c.mpc.thr_brake <= ct.throttle <= cs.throttle <= c.mpc.thr_max


def test_line_tracker_rejects_glitch_and_carries_line():
    from racer.mpc import LineTracker
    from racer.profile import MpcCfg
    from racer.vision import TrackEstimate
    lt = LineTracker(MpcCfg())
    good = TrackEstimate(valid=True, coeffs=np.array([0.0, 0.0, 0.0]), y_min=0.2, y_max=0.8)
    bad = TrackEstimate(valid=True, coeffs=np.array([0.3, 0.0, 0.0]), y_min=0.2, y_max=0.8)
    lt.update(good)
    lt.predict(1 / 60, 0.0, 0.0)
    line, gated = lt.update(bad)
    assert gated and np.all(np.abs(line[:, 0]) < 0.02)
    for _ in range(MpcCfg().max_coast + 1):          # persistent change is accepted eventually
        lt.predict(1 / 60, 0.0, 0.0)
        line, gated = lt.update(bad)
    assert not gated and np.median(line[:, 0]) > 0.2


def test_mpc_is_fast(prof):
    import time
    c = _mpc_cfg(prof)
    est = Vision(prof.vision, W, H).process(render(prof, [arc(-0.3, -0.7), arc(0.3, -1.3)]))
    ctrl = Controller(c)
    t = time.perf_counter()
    for i in range(100):
        ctrl.update(est, i / 60)
    assert (time.perf_counter() - t) / 100 < 0.012      # Pi 4 measured ~4.5 ms; margin for load
