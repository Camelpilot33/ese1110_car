"""Automatic settings sweep: one settle lap + N timed laps per combination.

Driven by the lap button (A at the start line). Pure logic, no hardware, so
it is unit-tested and the run loop only applies the settings it returns.

  sw = Sweep.parse("0.46:0.40,0.50:0.44", laps=5, settle=1)
  sw.current          -> {"speed_max": 0.46, "speed_min": 0.40}
  sw.on_lap(3.02)     -> LapRecord (timed or settle); advances when a combo is done
  sw.on_restart()     -> the next lap starts from standstill: it is not counted
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

KEYS = ("speed_max", "speed_min")


@dataclass
class LapRecord:
    duration: float
    combo: int
    settings: Dict[str, float]
    kind: str            # "timed" | "settle" | "restart" | "done"


@dataclass
class Sweep:
    combos: List[Dict[str, float]]
    laps: int = 5
    settle: int = 1
    index: int = 0
    _settle_left: int = field(default=-1)
    _timed: int = 0
    _skip_next: bool = False
    records: List[LapRecord] = field(default_factory=list)

    def __post_init__(self):
        if self._settle_left < 0:
            self._settle_left = self.settle

    @classmethod
    def parse(cls, spec: str, laps: int = 5, settle: int = 1) -> "Sweep":
        combos = []
        for part in spec.split(","):
            vals = [float(v) for v in part.strip().split(":")]
            if len(vals) != 2:
                raise ValueError(f"sweep entry must be speed_max:speed_min, got {part!r}")
            mx, mn = vals
            if mn > mx:
                raise ValueError(f"corner floor {mn} above straight speed {mx} in {part!r}")
            combos.append(dict(zip(KEYS, vals)))
        if not combos:
            raise ValueError("empty sweep")
        return cls(combos, laps, settle)

    @property
    def done(self) -> bool:
        return self.index >= len(self.combos)

    @property
    def current(self) -> Optional[Dict[str, float]]:
        return None if self.done else self.combos[self.index]

    def status(self) -> str:
        if self.done:
            return "sweep done"
        c = self.current
        phase = f"settle lap" if self._settle_left > 0 else f"timed lap {self._timed + 1}/{self.laps}"
        return (f"sweep {self.index + 1}/{len(self.combos)}: speed_max={c['speed_max']:.2f} "
                f"speed_min={c['speed_min']:.2f} ({phase})")

    def on_restart(self) -> None:
        """Car was stopped: the next lap includes a standing start."""
        self._skip_next = True

    def on_lap(self, duration: float) -> LapRecord:
        if self.done:
            rec = LapRecord(duration, len(self.combos) - 1, dict(self.combos[-1]), "done")
            self.records.append(rec)
            return rec
        settings = dict(self.current)
        if self._skip_next:
            self._skip_next = False
            kind = "restart"
        elif self._settle_left > 0:
            self._settle_left -= 1
            kind = "settle"
        else:
            self._timed += 1
            kind = "timed"
        rec = LapRecord(duration, self.index, settings, kind)
        self.records.append(rec)
        if kind == "timed" and self._timed >= self.laps:
            self.index += 1
            self._timed = 0
            self._settle_left = self.settle
        return rec

    def summary(self) -> List[dict]:
        out = []
        for i, c in enumerate(self.combos):
            t = [r.duration for r in self.records if r.combo == i and r.kind == "timed"]
            if t:
                mean = sum(t) / len(t)
                sd = (sum((x - mean) ** 2 for x in t) / len(t)) ** 0.5
                out.append({**c, "n": len(t), "mean": mean, "best": min(t), "sd": sd})
        return out

    def best(self) -> Optional[Dict[str, float]]:
        s = [r for r in self.summary() if r["n"] >= max(2, self.laps // 2)]
        if not s:
            return None
        b = min(s, key=lambda r: r["mean"])
        return {k: b[k] for k in KEYS}
