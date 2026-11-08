#!/usr/bin/env python3
"""Main loop.

  python run.py --profile profiles/<file>.yaml --mode manual --log
  python run.py --profile profiles/<file>.yaml --mode auto --stream

Gamepad (F710, X/XInput mode):
  RB (hold)   dead-man: motor runs only while held
  right stick X  steering (manual)    left stick Y  throttle (manual)
  X  toggle manual/auto    B  e-stop (latched)    START  clear e-stop
  A  mark a lap (press as the car crosses the start line)
  D-pad up/down     +/- 0.02 on speed_max (pid mode) or straight+turn throttle (live, logged)
  D-pad left/right  -/+ 0.02 speed_min / turn floor (pid mode) or -/+ 0.05 m lookahead (live, logged)
"""
from __future__ import annotations

import argparse
import signal
import statistics
import sys
import time

from racer import profile
from racer.control import Command, Controller
from racer.vision import Vision, draw_debug


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/default.yaml")
    ap.add_argument("--mode", choices=["manual", "auto"], default="manual")
    ap.add_argument("--log", action="store_true", help="record frames + state to logs/")
    ap.add_argument("--no-frames", action="store_true", help="with --log: state only")
    ap.add_argument("--stream", action="store_true", help="MJPEG debug view on :8080")
    ap.add_argument("--dry-run", action="store_true", help="mock actuators (wheels never move)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a profile value, e.g. control.lookahead_m=0.7")
    args = ap.parse_args(argv)

    prof = profile.load(args.profile, args.set)
    from racer.actuators import Actuators, Battery, MockActuators
    from racer.camera import PiCamera
    from racer.safety import Gamepad, SafetyGate

    act = MockActuators(prof.actuators) if args.dry_run else Actuators(prof.actuators)
    battery = None if args.dry_run else Battery(prof.actuators)
    cam = PiCamera(prof.camera)
    pad = Gamepad(prof.safety.joystick_device)
    gate = SafetyGate(prof.safety)
    vision = Vision(prof.vision, prof.camera.width, prof.camera.height)
    ctrl = Controller(prof.control)
    log = None
    if args.log:
        from racer.logger import RunLogger
        log = RunLogger("logs", args.mode, prof, save_frames=not args.no_frames)
        print(f"logging to {log.dir}")
    stream = None
    if args.stream:
        from tools.stream import MjpegServer
        stream = MjpegServer()

    running = True

    def _sigterm(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, _sigterm)

    mode = args.mode
    sc = prof.safety
    idx = -1
    loop_ms: list = []
    laps: list = []
    last_lap_t = None
    volts = battery.voltage() if battery else None
    last_batt_t = time.monotonic()
    last_print = 0.0
    if volts is not None and volts < sc.battery_min_v:
        print(f"WARNING: battery {volts:.2f} V < {sc.battery_min_v} V — charge before timing laps")
    print(f"mode={mode}  hold {sc.deadman_button.upper()} to drive, Ctrl-C to quit")

    try:
        while running:
            f = cam.get(idx, timeout=sc.frame_timeout_s)
            now = time.monotonic()
            if f is None:
                act.stop()                      # watchdog: no fresh frame
                continue
            img, t_frame, idx = f
            t0 = now

            for b in pad.pressed():
                if b == sc.estop_button:
                    gate.estop = True
                    print("E-STOP latched (START to clear)")
                elif b == "start" and gate.estop:
                    gate.estop = False
                    print("e-stop cleared")
                elif b == sc.mode_button:
                    mode = "auto" if mode == "manual" else "manual"
                    vision.reset()
                    print(f"mode -> {mode}")
                elif b in ("dpad_up", "dpad_down") and ctrl.cfg.mode == "pid":
                    d = 0.02 if b == "dpad_up" else -0.02
                    c = ctrl.cfg
                    c.speed_max = round(min(prof.actuators.throttle_max, max(c.speed_min, c.speed_max + d)), 3)
                    print(f"speed_max={c.speed_max} (min {c.speed_min}, cap {prof.actuators.throttle_max})")
                elif b in ("dpad_up", "dpad_down"):
                    d = 0.02 if b == "dpad_up" else -0.02
                    cap = prof.actuators.throttle_max
                    ctrl.cfg.straight_throttle = round(min(cap, max(0.0, ctrl.cfg.straight_throttle + d)), 3)
                    ctrl.cfg.turn_throttle = round(min(cap, max(0.0, ctrl.cfg.turn_throttle + d)), 3)
                    print(f"throttle straight={ctrl.cfg.straight_throttle} turn={ctrl.cfg.turn_throttle}"
                          f" (cap {prof.actuators.throttle_max})")
                elif b in ("dpad_left", "dpad_right") and ctrl.cfg.mode == "pid":
                    d = 0.02 if b == "dpad_right" else -0.02
                    c = ctrl.cfg
                    c.speed_min = round(min(c.speed_max, max(0.30, c.speed_min + d)), 3)
                    print(f"speed_min (turn floor)={c.speed_min}  speed_max={c.speed_max}")
                elif b in ("dpad_left", "dpad_right"):
                    d = 0.05 if b == "dpad_right" else -0.05
                    ctrl.cfg.lookahead_m = round(max(0.2, ctrl.cfg.lookahead_m + d), 3)
                    print(f"lookahead={ctrl.cfg.lookahead_m}")
                elif b == sc.lap_button:
                    if last_lap_t is not None:
                        laps.append(now - last_lap_t)
                        print(f"lap {len(laps)}: {laps[-1]:.3f} s  (best {min(laps):.3f})")
                    last_lap_t = now
                if log:
                    log.event("button", button=b, mode=mode,
                              straight_throttle=ctrl.cfg.straight_throttle, speed_max=ctrl.cfg.speed_max,
                              speed_min=ctrl.cfg.speed_min,
                              turn_throttle=ctrl.cfg.turn_throttle, lookahead_m=ctrl.cfg.lookahead_m)

            est = vision.process(img, keep_mask=stream is not None)
            if mode == "auto":
                cmd = ctrl.update(est, now)
            else:
                cmd = Command(steering=pad.axis("rjx"),
                              throttle=-pad.axis("ljy") * sc.manual_throttle_scale)

            ok = gate.allowed(now, pad.held(sc.deadman_button), pad.connected,
                              now - t_frame, mode == "auto")
            throttle = cmd.throttle if ok else 0.0
            act.set(cmd.steering, throttle)

            if battery and now - last_batt_t > 1.0:
                volts, last_batt_t = battery.voltage(), now
            dt_ms = (time.monotonic() - t0) * 1000
            latency_ms = (time.monotonic() - t_frame) * 1000
            loop_ms.append(dt_ms)

            if log:
                log.log({"t": now, "frame": idx, "mode": mode, "steering": round(cmd.steering, 4),
                         "throttle": round(throttle, 4), "cmd_throttle": round(cmd.throttle, 4),
                         "ok": ok, "why": gate.reasons, "phase": cmd.phase, "lost": cmd.lost,
                         "kappa": round(cmd.kappa_preview, 3), "turn_f": round(cmd.turn_factor, 3), "err": round(cmd.error, 4), "target": cmd.target,
                         "loop_ms": round(dt_ms, 2), "latency_ms": round(latency_ms, 2),
                         "battery_v": volts, "vision": est.summary()}, img)
            if stream:
                txt = f"{mode} {'GO' if ok else ','.join(gate.reasons)} s={cmd.steering:+.2f} t={throttle:+.2f} {latency_ms:.0f}ms"
                try:
                    stream.publish(draw_debug(img, vision, est, cmd.target, txt))
                except Exception as e:  # debug view must never stop the car loop
                    print(f"stream error: {e}")
            if now - last_print > 2.0:
                last_print = now
                recent = loop_ms[-100:]
                print(f"[{mode}] loop {statistics.mean(recent):.1f} ms (max {max(recent):.1f})  "
                      f"latency {latency_ms:.0f} ms  valid={est.valid}  "
                      f"batt={volts if volts is None else round(volts, 2)}V  "
                      f"{'GO' if ok else ','.join(gate.reasons)}")
    except KeyboardInterrupt:
        pass
    finally:
        act.close()
        cam.close()
        pad.close()
        if stream:
            stream.close()
        if log:
            log.close()
            print(f"log: {log.dir}  (dropped {log.dropped} frames)")
        if loop_ms:
            print(f"loop ms: mean {statistics.mean(loop_ms):.1f}  max {max(loop_ms):.1f}")
        c = ctrl.cfg
        print(f"final: mode={c.mode} speed_max={c.speed_max} speed_min={c.speed_min} straight_throttle={c.straight_throttle} "
              f"turn_throttle={c.turn_throttle} lookahead_m={c.lookahead_m}  (copy into the profile to keep)")
        if laps:
            print(f"laps: {len(laps)}  best {min(laps):.3f}  "
                  f"mean of last 5 {statistics.mean(laps[-5:]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
