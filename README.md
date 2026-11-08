# PiRacer Pro — oval racer

Classical control for the small oval: orange-tape detection → pure pursuit →
straight/turn speed profile. Plain Python (picamera2 + OpenCV), no ROS, no
DonkeyCar at runtime. Every tuning value lives in `profiles/*.yaml`.

## The car (surveyed 2026-11-07)

| | |
|---|---|
| Board | Raspberry Pi 4B, 4 GB, Debian 12 bookworm, Python 3.11 |
| Host | `ssh ese1110` → `pi@10.103.211.129` |
| Python env | `~/env` (venv with system site-packages: picamera2, OpenCV 4.11, numpy 1.26, PyYAML, smbus2) |
| Camera | OV5647 with a **wide-angle lens (strong barrel distortion)**, mounted upside down (hflip+vflip), low and close to horizontal. 640×480 sensor mode = full FoV at ~59 fps; 320×240 output measured at 63 fps |
| PWM | PCA9685 at I2C 0x40, ch0 = steering servo, ch1 = ESC, 60 Hz |
| Battery | INA219 at 0x42, 2S pack (6.0–8.4 V); 8.24 V at survey |
| OLED | SSD1306 at 0x3c, driven by `picard_display.service` (shows IP, battery) |
| Gamepad | Logitech F710 (XInput mode) on `/dev/input/js0` |
| Track | silver crinkled tarp (reflective) with **orange tape** boundary lines |
| Old stack | DonkeyCar 5.3 in `~/mycar` (behaviour cloning). Its PWM calibration was copied into `profiles/default.yaml` |

DonkeyCar calibration (ticks at 60 Hz) → profile (µs): steer left 500→2035,
right 330→1343, throttle stop 370→1506, forward 500→2035, reverse 320→1302.

## Layout

```
racer/        camera, vision, control, actuators, safety, logger, geometry, profile
tools/        calibrate.py  replay.py  stream.py  ground_cal.py  lens_cal.py  actuator_cal.py
profiles/     default.yaml + one <date>_<venue>.yaml per session
run.py        main loop
logs/         gitignored
tests/        pytest, synthetic images, runs on the laptop
```

## Setup

Laptop (tests + replay):

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/python -m pytest -q
```

Pi: nothing to install, `~/env` already has everything. The repo lives at
`~/piracer-oval`; push from the laptop with:

```bash
git push pi main
```

(The Pi repo uses `receive.denyCurrentBranch=updateInstead`, so a push updates the files there directly.)

On the Pi:

```bash
cd ~/piracer-oval && source ~/env/bin/activate
```

## Gamepad

| Control | Action |
|---|---|
| **RB (hold)** | dead-man. Motor runs only while it is held |
| right stick X / left stick Y | steering / throttle (manual mode) |
| X | toggle manual ↔ auto |
| B | e-stop (latched); START clears it |
| A | lap mark (press at the start line). `run.py` prints lap times |
| D-pad up/down | pid mode: `speed_max` ±0.02 (pure pursuit: straight+turn throttle) |
| D-pad left/right | pid mode: `speed_min` (turn floor) ±0.02 (pure pursuit: lookahead ±0.05) |

The motor also stops if no new camera frame arrives within `safety.frame_timeout_s` (100 ms),
if the gamepad disconnects, or after `safety.max_run_s` in auto (0 = off).

## One-time camera setup (per car, not per venue)

1. **Lens**: record a printed chessboard, then
   `python tools/lens_cal.py profiles/default.yaml "logs/<run>/frames/*.jpg" --board 9x6`.
   Until this is done the lens values are guesses.
2. **Ground plane**: tape 4+ markers on the floor, measure each from the rear-axle
   centre (x right, y forward, metres), read their pixels in `tools/stream.py --raw`, then
   `python tools/ground_cal.py profiles/default.yaml --pt U,V:X,Y ...` (≥ 4 `--pt`).
   Until then, distances come from a guessed camera pose (`vision.geometry`).
3. Measure `vision.track_width_m` (between tape centres) and `control.wheelbase_m`.
4. Steering trim: `python tools/actuator_cal.py` with the car on a stand.

## At the venue

```bash
python tools/calibrate.py --venue <name>          # lock exposure/WB, sample tape colour
python tools/stream.py --profile profiles/<file>.yaml    # check overlay at http://<pi>:8080
python run.py --profile profiles/<file>.yaml --mode manual --log   # a few manual laps
python tools/replay.py logs/<run> --video out.mp4 --set vision.roi_top=0.42   # tune offline (laptop: scp the log)
python run.py --profile profiles/<file>.yaml --mode auto --stream  # tune, then timed run
```

Tuning order (one change at a time, 5 laps each, keep only if mean lap time drops
without crashes): steering/lookahead at low constant speed → `turn_throttle` from
the circle test (85–90 % of slip) → raise `straight_throttle` and move
`brake_preview_m` / `brake_throttle` / `brake_time_s` until turn entry speed is right →
raise `lookahead_m` only if the car oscillates. Raise `actuators.throttle_max` (hard cap,
default 0.35) once the car is stable.

Use `--set key=value` on any script to try a value without editing the profile.

## Progress (lab oval, 2026-11-07)

| Run | Controller | Laps | Best | Mean | Longest stretch |
|---|---|---|---|---|---|
| 3 | pure pursuit, fitted geometry | 19 | 3.88 s | 4.12 s | 94 s |
| 6 | PID + speed output, lookahead fix | 23 | 3.29 s | 3.54 s | 151 s |

## Notes

- Dim room: auto exposure picked 33 ms. That blurs badly at speed. `calibrate.py` caps
  exposure (`--max-exposure-us`, default 8 ms) and raises gain to compensate. More light
  at the venue helps.
- The tarp is reflective. HSV thresholds on the orange tape ignore it; brightness-based edge detection would not.
- Curvature for turn detection comes from a circle fit on the centre points. Its magnitude
  is only approximate, so set `turn_kappa_enter/exit` from replay output, not physics.
- Steering convention: −1 = left, +1 = right. Ground frame: x right, y forward, origin at the rear axle.
