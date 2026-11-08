#!/usr/bin/env python3
"""Main loop.

  python run.py --profile profiles/<file>.yaml --mode manual --log
  python run.py --profile profiles/<file>.yaml --mode auto --stream

Gamepad (F710, X/XInput mode):
  RB (hold)   dead-man: motor runs only while held
  right stick X  steering (manual)    left stick Y  throttle (manual)
  X  toggle manual/auto    B  e-stop (latched)    START  clear e-stop
  A  mark a lap (press as the car crosses the start line); a lap in which the car was
     stopped is not counted
  --sweep MAX:MIN,...  automatic speed sweep (pid): 1 settle + 5 timed laps per combination
  D-pad up/down     +/- 0.02 on speed_max (pid mode) or straight+turn throttle (live, logged)
  D-pad left/right  -/+ 0.02 speed_min / turn floor (pid mode) or -/+ 0.05 m lookahead (live, logged)
  mpc mode: D-pad up/down = cornering grip a_lat +/-0.25, right/left = tightness w_len x1.25 / x0.8
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
    ap.add_argument("--sweep", metavar="MAX:MIN,...",
                    help="pid mode: automatic speed sweep, e.g. 0.46:0.40,0.50:0.44 "
                         "(speed_max:speed_min per combination; A = lap)")
    ap.add_argument("--sweep-laps", type=int, default=5, help="timed laps per combination")
    ap.add_argument("--sweep-settle", type=int, default=1, help="uncounted laps after each change")
    args = ap.parse_args(argv)

    prof = profile.load(args.profile, args.set)
    from racer.actuators import Actuators, Battery, MockActuators, steer_limits
    prof.control.steer_min, prof.control.steer_max = steer_limits(prof.actuators)
    prof.control.mpc.half_width_m = prof.vision.track_width_m / 2
    from racer.camera import PiCamera
    from racer.safety import Gamepad, SafetyGate

    act = MockActuators(prof.actuators) if args.dry_run else Actuators(prof.actuators)
    battery = None if args.dry_run else Battery(prof.actuators)
    cam = PiCamera(prof.camera)
    pad = Gamepad(prof.safety.joystick_device)
    gate = SafetyGate(prof.safety)
    vision = Vision(prof.vision, prof.camera.width, prof.camera.height)
    ctrl = Controller(prof.control)
    sweep = None
    if args.sweep:
        from racer.sweep import Sweep
        if prof.control.mode != "pid":
            raise SystemExit("--sweep needs control.mode pid")
        sweep = Sweep.parse(args.sweep, args.sweep_laps, args.sweep_settle)
        for k, v in sweep.current.items():
            setattr(ctrl.cfg, k, v)
        print(f"SWEEP: {len(sweep.combos)} combinations x ({args.sweep_settle} settle + {args.sweep_laps} timed) laps."
              f" Press A at the start line every lap; D-pad speed is locked.")
        print(sweep.status())
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
    was_ok = False
    sc = prof.safety
    idx = -1
    loop_ms: list = []
    laps: list = []
    last_lap_t = None
    lap_dirty = False              # car stopped during this lap: it is not a real lap time
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
                elif b.startswith("dpad") and sweep is not None and not sweep.done:
                    print(f"D-pad locked during the sweep ({sweep.status()})")
                elif b in ("dpad_up", "dpad_down") and ctrl.cfg.mode == "mpc":
                    m = ctrl.cfg.mpc
                    m.a_lat = round(max(1.0, m.a_lat + (0.25 if b == "dpad_up" else -0.25)), 2)
                    print(f"mpc a_lat (cornering grip)={m.a_lat}")
                elif b in ("dpad_left", "dpad_right") and ctrl.cfg.mode == "mpc":
                    m = ctrl.cfg.mpc
                    m.w_len = round(min(8000.0, max(50.0, m.w_len * (1.25 if b == "dpad_right" else 0.8))), 1)
                    print(f"mpc w_len (tightness)={m.w_len}  (right = tighter line in turns)")
                elif b in ("dpad_up", "dpad_down") and ctrl.cfg.mode == "pid":
                    d = 0.02 if b == "dpad_up" else -0.02
                    c = ctrl.cfg
                    new_v = round(min(prof.actuators.throttle_max, max(c.speed_min, c.speed_max + d)), 3)
                    if new_v == c.speed_max:
                        why = (f"cap {prof.actuators.throttle_max}" if d > 0 else f"corner floor {c.speed_min}")
                        print(f"speed_max AT LIMIT {c.speed_max} ({why}) - press ignored")
                    c.speed_max = new_v
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
                    new_v = round(min(c.speed_max, max(0.30, c.speed_min + d)), 3)
                    if new_v == c.speed_min:
                        why = (f"cannot exceed speed_max {c.speed_max} - raise it with D-pad up first"
                               if d > 0 else "minimum 0.30")
                        print(f"speed_min AT LIMIT {c.speed_min} ({why}) - press ignored")
                    c.speed_min = new_v
                    print(f"speed_min (turn floor)={c.speed_min}  speed_max={c.speed_max}")
                elif b in ("dpad_left", "dpad_right"):
                    d = 0.05 if b == "dpad_right" else -0.05
                    ctrl.cfg.lookahead_m = round(max(0.2, ctrl.cfg.lookahead_m + d), 3)
                    print(f"lookahead={ctrl.cfg.lookahead_m}")
                elif b == sc.lap_button:
                    if last_lap_t is not None:
                        dur = now - last_lap_t
                        kind = "restart" if lap_dirty else "timed"
                        settings = {"speed_max": ctrl.cfg.speed_max, "speed_min": ctrl.cfg.speed_min}
                        if sweep is not None:
                            if lap_dirty:
                                sweep.on_restart()
                            rec = sweep.on_lap(dur)
                            kind, settings = rec.kind, rec.settings
                        if kind == "timed" or (sweep is None and kind != "restart"):
                            laps.append(dur)
                        tag = "" if kind == "timed" else f"  [{kind}, not counted]"
                        print(f"lap: {dur:.3f} s{tag}  (best {min(laps):.3f})" if laps else f"lap: {dur:.3f} s{tag}")
                        if log:
                            log.event("lap", duration=round(dur, 4), kind=kind,
                                      combo=None if sweep is None else (sweep.records[-1].combo if sweep.records else None),
                                      **settings)
                        if sweep is not None:
                            if sweep.done:
                                best = sweep.best()
                                if best and not getattr(sweep, "_announced", False):
                                    for k, v in best.items():
                                        setattr(ctrl.cfg, k, v)
                                    sweep._announced = True
                                    print(f"SWEEP DONE - driving on with the fastest combination {best}")
                            else:
                                for k, v in sweep.current.items():
                                    setattr(ctrl.cfg, k, v)
                                print(sweep.status())
                    last_lap_t = now
                    lap_dirty = False
                if log:
                    log.event("button", button=b, mode=mode,
                              straight_throttle=ctrl.cfg.straight_throttle, speed_max=ctrl.cfg.speed_max,
                              speed_min=ctrl.cfg.speed_min, a_lat=ctrl.cfg.mpc.a_lat, w_len=ctrl.cfg.mpc.w_len,
                              turn_throttle=ctrl.cfg.turn_throttle, lookahead_m=ctrl.cfg.lookahead_m)

            ok = gate.allowed(now, pad.held(sc.deadman_button), pad.connected,
                              now - t_frame, mode == "auto")
            if was_ok and not ok:
                lap_dirty = True           # stopped mid-lap: this lap's time is not real
            if ok and not was_ok and mode == "auto":
                # Fresh start (dead-man pressed): forget stale state and
                # soft-start, so the car settles on the line before speeding up.
                vision.reset()
                ctrl.on_start(now)
                if log:
                    log.event("start", mode=mode)
            was_ok = ok

            est = vision.process(img, keep_mask=stream is not None)
            if mode == "auto":
                cmd = ctrl.update(est, now)
            else:
                cmd = Command(steering=pad.axis("rjx"),
                              throttle=-pad.axis("ljy") * sc.manual_throttle_scale)

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
                         "mpc": cmd.info, "loop_ms": round(dt_ms, 2), "latency_ms": round(latency_ms, 2),
                         "battery_v": volts, "vision": est.summary()}, img)
            if stream:
                txt = f"{mode} {'GO' if ok else ','.join(gate.reasons)} s={cmd.steering:+.2f} t={throttle:+.2f} {latency_ms:.0f}ms"
                try:
                    if cmd.info:
                        txt += f" v{cmd.info['v_est']:.1f}>{cmd.info['v_target']:.1f}"
                    stream.publish(draw_debug(img, vision, est, cmd.target, txt, plan=cmd.plan))
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
        if sweep is not None and sweep.summary():
            print("sweep results (timed laps only):")
            for r in sweep.summary():
                print(f"  speed_max {r['speed_max']:.2f} speed_min {r['speed_min']:.2f}: "
                      f"n={r['n']} mean {r['mean']:.3f} best {r['best']:.3f} sd {r['sd']:.3f}")
        if laps:
            print(f"laps: {len(laps)}  best {min(laps):.3f}  "
                  f"mean of last 5 {statistics.mean(laps[-5:]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
