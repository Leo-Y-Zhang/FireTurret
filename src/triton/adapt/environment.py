# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Where air density and wind come from.

Three providers, all satisfying one protocol:

    NullEnvironment    standard air, no wind. The default, and bit-identical
                       to having no environment layer at all.
    FileEnvironment    reads a JSON file a weather station writes. Also the
                       test double, which is deliberate — a test double that
                       is also a shipped feature gets exercised for real.
    SerialEnvironment  a sensor pod on a second serial port (pyserial is
                       already a core dependency, so this adds nothing).

## The density correction is exact, not fitted

The quadratic drag constant is

    k = rho * C_d * A / (2 m)

so it is *linear* in air density. Correcting for density is therefore a
multiplication by the density ratio, with no fitting and no new parameter:

    k_eff = k * (rho / rho_ref)

That matters on a hot day or at altitude. Computed, not estimated: 35 C at
84 kPa (roughly 1500 m on a hot day) gives a density ratio of **0.775** — about
22 percent less drag, which at a 7 m throw is a real range change rather than a
rounding error.

Air density comes from the ideal gas law with a humidity correction via the Buck
equation, because moist air is *less* dense than dry air at the same temperature
and pressure — which surprises people often enough to be worth stating.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# Sea-level standard: 15 C, 101325 Pa, dry.
RHO_REF = 1.225  # kg/m^3
R_DRY = 287.058  # J/(kg K)
R_VAPOUR = 461.495  # J/(kg K)


@dataclass(frozen=True)
class Environment:
    """Ambient conditions. `wind_*` is the WORLD-frame wind vector in m/s."""

    temperature_c: float = 15.0
    pressure_pa: float = 101325.0
    relative_humidity: float = 0.0  # 0..1
    wind_east_ms: float = 0.0
    wind_north_ms: float = 0.0
    anemometer: bool = False  # is the wind MEASURED, or merely assumed zero?

    def __post_init__(self) -> None:
        # Validate at CONSTRUCTION, not at first use. These values arrive from a
        # JSON file a weather station wrote, so a string where a float belongs is
        # a realistic input — and it must fail here, where the providers' own
        # error handling can degrade to STANDARD, rather than surfacing later as
        # a TypeError deep inside a ballistics integration.
        for field_name in ("temperature_c", "pressure_pa", "relative_humidity",
                           "wind_east_ms", "wind_north_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number, got {value!r}")
        if self.pressure_pa <= 0:
            raise ValueError("pressure must be positive")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative humidity must be within [0, 1]")
        if self.temperature_c <= -273.15:
            raise ValueError("temperature below absolute zero")

    @property
    def air_density(self) -> float:
        """Density of moist air, kg/m^3."""
        t_k = self.temperature_c + 273.15
        p_vapour = saturation_vapour_pressure_pa(self.temperature_c) * self.relative_humidity
        p_dry = self.pressure_pa - p_vapour
        return p_dry / (R_DRY * t_k) + p_vapour / (R_VAPOUR * t_k)

    @property
    def density_ratio(self) -> float:
        return self.air_density / RHO_REF

    def effective_drag_k(self, drag_k: float) -> float:
        """`k` scales linearly with density — exact, not fitted."""
        return drag_k * self.density_ratio

    @property
    def wind_speed_ms(self) -> float:
        return math.hypot(self.wind_east_ms, self.wind_north_ms)


STANDARD = Environment()


def saturation_vapour_pressure_pa(temperature_c: float) -> float:
    """Buck equation. Accurate to ~0.05% over the range a turret operates in,
    which is far better than the humidity reading feeding it."""
    t = temperature_c
    return 611.21 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


@runtime_checkable
class EnvironmentProvider(Protocol):
    def read(self) -> Environment: ...


class NullEnvironment:
    """Standard air, no wind, and `anemometer=False` so downstream layers know
    the wind is ASSUMED rather than measured. That distinction is load-bearing
    in `adapt/wind.py`."""

    def read(self) -> Environment:
        return STANDARD


class FileEnvironment:
    """Reads a JSON file a weather station writes.

    Missing file, malformed JSON and unknown keys all degrade to STANDARD rather
    than raising: an environment source is an optional convenience, and a turret
    must not fail to run because a weather station rebooted.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> Environment:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return STANDARD
        if not isinstance(raw, dict):
            return STANDARD
        allowed = {
            "temperature_c", "pressure_pa", "relative_humidity",
            "wind_east_ms", "wind_north_ms", "anemometer",
        }
        try:
            return Environment(**{k: v for k, v in raw.items() if k in allowed})
        except (TypeError, ValueError):
            return STANDARD


class SerialEnvironment:
    """A sensor pod on a second serial port, speaking one line of JSON per read.

    The port is opened lazily and every failure degrades to STANDARD, for the
    same reason as `FileEnvironment`: this is a convenience, not a dependency.
    """

    def __init__(self, port: str, baud: int = 9600, timeout_s: float = 0.5,
                 serial_factory=None) -> None:
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self._factory = serial_factory
        self._serial = None

    def _open(self):
        if self._serial is not None:
            return self._serial
        if self._factory is not None:
            self._serial = self._factory(self.port, self.baud, timeout=self.timeout_s)
        else:  # pragma: no cover - requires hardware
            import serial

            self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        return self._serial

    def read(self) -> Environment:
        try:
            line = self._open().readline()
        except Exception:  # noqa: BLE001 - any link failure degrades, never raises
            return STANDARD
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        try:
            raw = json.loads(line)
            allowed = {
                "temperature_c", "pressure_pa", "relative_humidity",
                "wind_east_ms", "wind_north_ms", "anemometer",
            }
            return Environment(**{k: v for k, v in raw.items() if k in allowed})
        except (TypeError, ValueError):
            return STANDARD
