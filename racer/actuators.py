"""Steering servo + ESC through the PiRacer's PCA9685 (I2C 0x40), plus the
INA219 battery monitor (0x42). Talks to the chips directly with smbus2 so no
DonkeyCar/Adafruit packages are needed.
"""
from __future__ import annotations

import time
from typing import Optional

from .profile import ActuatorCfg

_MODE1, _MODE2, _PRESCALE, _LED0_ON_L = 0x00, 0x01, 0xFE, 0x06
_RESTART, _SLEEP, _AI, _ALLCALL, _OUTDRV = 0x80, 0x10, 0x20, 0x01, 0x04
_OSC_HZ = 25_000_000


def _interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def steering_pulse_us(cfg: ActuatorCfg, steering: float) -> float:
    s = max(-1.0, min(1.0, steering))
    if s < 0:
        return _interp(s, 0, -1, cfg.steering_center_us, cfg.steering_left_us)
    return _interp(s, 0, 1, cfg.steering_center_us, cfg.steering_right_us)


def throttle_pulse_us(cfg: ActuatorCfg, throttle: float) -> float:
    t = max(-cfg.reverse_max, min(cfg.throttle_max, throttle))
    if t >= 0:
        return _interp(t, 0, 1, cfg.throttle_stop_us, cfg.throttle_forward_us)
    return _interp(t, 0, -1, cfg.throttle_stop_us, cfg.throttle_reverse_us)


class PCA9685:
    def __init__(self, bus, addr: int, freq_hz: float):
        self.bus, self.addr = bus, addr
        self.freq_hz = freq_hz
        # Same rounding as Adafruit_PCA9685 (used by DonkeyCar), so tick values match.
        prescale = int(_OSC_HZ / 4096.0 / freq_hz - 1.0 + 0.5)
        bus.write_byte_data(addr, _MODE2, _OUTDRV)
        bus.write_byte_data(addr, _MODE1, _ALLCALL)
        time.sleep(0.005)
        old = bus.read_byte_data(addr, _MODE1) & ~_SLEEP
        bus.write_byte_data(addr, _MODE1, (old & 0x7F) | _SLEEP)
        bus.write_byte_data(addr, _PRESCALE, prescale)
        bus.write_byte_data(addr, _MODE1, old)
        time.sleep(0.005)
        bus.write_byte_data(addr, _MODE1, old | _RESTART | _AI)

    def set_pulse_us(self, channel: int, us: float) -> int:
        ticks = int(round(us * 4096 * self.freq_hz / 1e6))
        ticks = max(0, min(4095, ticks))
        self.bus.write_i2c_block_data(self.addr, _LED0_ON_L + 4 * channel,
                                      [0, 0, ticks & 0xFF, ticks >> 8])
        return ticks

    def off(self, channel: int) -> None:
        self.bus.write_i2c_block_data(self.addr, _LED0_ON_L + 4 * channel, [0, 0, 0, 0])


class Actuators:
    def __init__(self, cfg: ActuatorCfg, bus=None):
        if bus is None:
            from smbus2 import SMBus
            bus = SMBus(cfg.i2c_bus)
        self.cfg = cfg
        self.bus = bus
        self.pca = PCA9685(bus, cfg.pca9685_addr, cfg.pwm_freq_hz)
        self.steering = 0.0
        self.throttle = 0.0
        self.stop()

    def set(self, steering: float, throttle: float) -> None:
        self.steering, self.throttle = steering, throttle
        self.pca.set_pulse_us(self.cfg.steering_channel, steering_pulse_us(self.cfg, steering))
        self.pca.set_pulse_us(self.cfg.throttle_channel, throttle_pulse_us(self.cfg, throttle))

    def stop(self) -> None:
        self.set(self.steering, 0.0)

    def close(self) -> None:
        try:
            self.set(0.0, 0.0)
            time.sleep(0.05)
        finally:
            self.pca.off(self.cfg.steering_channel)


class MockActuators:
    """Stand-in for laptop runs and tests."""
    def __init__(self, cfg: ActuatorCfg, *_, **__):
        self.cfg = cfg
        self.steering = self.throttle = 0.0

    def set(self, steering: float, throttle: float) -> None:
        self.steering = max(-1.0, min(1.0, steering))
        self.throttle = max(-self.cfg.reverse_max, min(self.cfg.throttle_max, throttle))

    def stop(self) -> None:
        self.throttle = 0.0

    def close(self) -> None:
        self.set(0.0, 0.0)


class Battery:
    """INA219 bus voltage (no calibration register needed for voltage)."""
    def __init__(self, cfg: ActuatorCfg, bus=None):
        if bus is None:
            from smbus2 import SMBus
            bus = SMBus(cfg.i2c_bus)
        self.bus, self.addr = bus, cfg.ina219_addr

    def voltage(self) -> Optional[float]:
        try:
            d = self.bus.read_i2c_block_data(self.addr, 0x02, 2)
            return ((d[0] << 8 | d[1]) >> 3) * 0.004
        except OSError:
            return None
