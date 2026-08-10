# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Water-jet ballistics: pump pressure → exit velocity, drag-corrected arc
integration, elevation solving, and pressure selection.

The model is deliberately simple and calibratable: a discharge coefficient maps
pressure to exit speed (Bernoulli), and a single quadratic-drag constant models
jet breakup. `fireturret fit` refines both from real test shots; the closed-loop
servo absorbs whatever error remains.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .config import JetConfig

G = 9.81
WATER_DENSITY = 1000.0  # kg/m^3
PSI_TO_PA = 6894.76
_DT = 0.004
_MAX_STEPS = 6000


def exit_velocity(pump_pct: float, cfg: JetConfig) -> float:
    """Jet exit speed (m/s) for a pump command in percent."""
    pct = max(0.0, min(100.0, pump_pct))
    pressure_pa = (pct / 100.0) * cfg.max_pressure_psi * PSI_TO_PA
    return cfg.velocity_coeff * math.sqrt(2.0 * pressure_pa / WATER_DENSITY)


@dataclass(frozen=True)
class ArcResult:
    """A full arc integration: geometry AND the timing that goes with it.

    `simulate_arc` returned only the points, which is why the suppression loop
    had no way to know its own transport delay — the water it observes was
    launched most of a second before the correction it is being credited to
    (spec §3.5). Three source designs each wanted a different incompatible
    extension to `simulate_arc`; this is the one shape that serves all of them,
    and `simulate_arc` stays a thin wrapper so nothing existing has to change.
    """

    points: list[tuple[float, float]]
    flight_time_s: float
    impact_speed_mps: float

    @property
    def range_m(self) -> float:
        return self.points[-1][0]


def arc_with_time(
    speed: float,
    elevation_deg: float,
    cfg: JetConfig,
    wind: tuple[float, float] = (0.0, 0.0),
    drag_k: float | None = None,
) -> ArcResult:
    """Integrate the jet arc (RK4, quadratic drag) from the nozzle until it lands.

    The final step is interpolated to the ground, so `flight_time_s` includes the
    fractional last step rather than rounding down to a whole `_DT` — the
    difference is small but it is a systematic bias, and this number is about to
    become load-bearing for the settle window.
    """
    theta = math.radians(elevation_deg)
    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)
    x, y = 0.0, cfg.nozzle_height_m
    pts = [(x, y)]

    k = cfg.drag_k if drag_k is None else drag_k
    wx, wy = wind

    def accel(vx_: float, vy_: float) -> tuple[float, float]:
        # Drag acts on AIRSPEED, not groundspeed: a = -g - k|v - w|(v - w).
        # With w = (0, 0) this is bit-identical to the groundspeed form it
        # replaces, which `tests/test_environment.py` asserts — the wind term is
        # a genuine extension, not a re-derivation of the existing physics.
        rx, ry = vx_ - wx, vy_ - wy
        v = math.hypot(rx, ry)
        return (-k * v * rx, -G - k * v * ry)

    steps = 0
    for _ in range(_MAX_STEPS):
        # RK4 on (x, y, vx, vy)
        a1x, a1y = accel(vx, vy)
        k1 = (vx, vy, a1x, a1y)
        a2x, a2y = accel(vx + a1x * _DT / 2, vy + a1y * _DT / 2)
        k2 = (vx + a1x * _DT / 2, vy + a1y * _DT / 2, a2x, a2y)
        a3x, a3y = accel(vx + a2x * _DT / 2, vy + a2y * _DT / 2)
        k3 = (vx + a2x * _DT / 2, vy + a2y * _DT / 2, a3x, a3y)
        a4x, a4y = accel(vx + a3x * _DT, vy + a3y * _DT)
        k4 = (vx + a3x * _DT, vy + a3y * _DT, a4x, a4y)

        nx = x + (_DT / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ny = y + (_DT / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        vx = vx + (_DT / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        vy = vy + (_DT / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        if ny <= 0.0:
            t = y / (y - ny) if y != ny else 1.0
            pts.append((x + (nx - x) * t, 0.0))
            return ArcResult(
                points=pts,
                flight_time_s=(steps + t) * _DT,
                impact_speed_mps=math.hypot(vx, vy),
            )
        x, y = nx, ny
        steps += 1
        pts.append((x, y))
    return ArcResult(
        points=pts,
        flight_time_s=steps * _DT,
        impact_speed_mps=math.hypot(vx, vy),
    )


def simulate_arc(
    speed: float,
    elevation_deg: float,
    cfg: JetConfig,
) -> list[tuple[float, float]]:
    """The arc's geometry. Unchanged signature and unchanged return value — a
    thin wrapper over `arc_with_time` so every existing caller is untouched."""
    return arc_with_time(speed, elevation_deg, cfg).points


# Cached search tables, one per JetConfig. Keyed by the config's own values
# rather than by identity, so two equal configs share a table and a mutated one
# gets a fresh table instead of a stale hit.
_RANGE_TABLES: dict[tuple, object] = {}


def _range_table(cfg: JetConfig):
    """The cached search table for this jet. Built once, ~25x25 RK4 arcs."""
    from .control.ranging import SpeedRangeTable

    key = (cfg.velocity_coeff, cfg.drag_k, cfg.max_pressure_psi,
           cfg.nozzle_height_m, cfg.min_pump_pct)
    table = _RANGE_TABLES.get(key)
    if table is None:
        table = SpeedRangeTable(cfg)
        _RANGE_TABLES[key] = table
    return table


def range_of(speed: float, elevation_deg: float, cfg: JetConfig) -> float:
    """Horizontal landing distance for a given exit speed and elevation."""
    return simulate_arc(speed, elevation_deg, cfg)[-1][0]


def flight_time_of(speed: float, elevation_deg: float, cfg: JetConfig) -> float:
    """Time of flight from nozzle to ground, in seconds.

    Water is airborne for ~0.94 s at minimum coherent pump and ~1.28 s at full
    pump on the default low arc — far longer than any control cycle, which is
    why anything that must wait for water to LAND (a keep-out transit hold, a
    settle window) has to be derived from this rather than from a frame count.
    """
    return arc_with_time(speed, elevation_deg, cfg).flight_time_s


def _search_range_of(speed: float, elevation_deg: float, cfg: JetConfig) -> float:
    """Range, for use inside a SEARCH loop.

    Backed by a cached bilinear table (`control.ranging.RangeTable`) rather than a
    fresh RK4 integration. The distinction from `range_of` is deliberate and the
    boundary matters:

    - **searching** — `peak_elevation`'s grid scan and `solve_elevation`'s
      bisection call this hundreds of times per solution to locate a root. They
      need shape, not last-digit accuracy.
    - **reporting** — `FiringSolution.predicted_range_m` still uses the exact
      `range_of` at the chosen elevation, so the number handed to the rest of the
      system is never an interpolation.

    Measured on an idle machine: `plan_suppression` **67.3 ms -> 0.606 ms**, a
    111x speedup, for a chosen elevation differing by at most **0.036 deg**.
    That is 7.5 mm of range at 7.2 m, and 1.8% of the tilt slew the turret
    covers in a single tick — far below the 12 px range deadband. Pan, pump and
    every valve, laser and warn decision are bit-identical.

    Why it matters rather than being a micro-optimisation: the exact version
    **exceeds the 33.3 ms tick budget** (67.3 ms idle here; 43.7 ms when first
    characterised — either way over), so a RANGE tick was already an overrun
    that `runner.py`'s deadline watchdog would flag. SP8's moving platform
    re-plans continuously rather than once per engagement, which turns an
    occasional overrun into a permanent one.

    ## Outside the table, integrate exactly

    The table spans the elevations and speeds the turret can actually command.
    Callers are not required to stay inside it: `solve_elevation` accepts an
    explicit `max_elevation_deg`, and a high-arc (lobbing) solution legitimately
    asks about 80 deg.

    Interpolants clamp at their edge, so without this branch such a query would
    silently return the range at 60 deg — a plausible number, badly wrong, with
    nothing to indicate it. That is the characteristic way a lookup table
    introduces a bug: not by being imprecise, but by answering a question it was
    not built for. Out-of-grid queries therefore fall through to exact
    integration. They are rare, so the cost is not on the hot path, and the
    table's extent stays an optimisation detail rather than a limit on what the
    solver can be asked.
    """
    table = _range_table(cfg)
    if not table.covers(speed, elevation_deg):
        return range_of(speed, elevation_deg, cfg)
    return table.range_of_speed(speed, elevation_deg)


def peak_elevation(
    speed: float,
    cfg: JetConfig,
    min_elevation_deg: float,
    max_elevation_deg: float,
) -> float:
    """Elevation of maximum range. With quadratic drag the optimum sits well
    below the vacuum 45° (typically ~30° for a water jet) — coarse grid then
    ternary refinement over the allowed band.
    """
    lo = min_elevation_deg
    hi = max_elevation_deg
    if hi <= lo:
        return lo
    best_e = lo
    best_r = -1.0
    steps = max(2, int((hi - lo) / 2.0))
    for i in range(steps + 1):
        e = lo + (hi - lo) * i / steps
        r = _search_range_of(speed, e, cfg)
        if r > best_r:
            best_r = r
            best_e = e
    a = max(lo, best_e - 2.0)
    b = min(hi, best_e + 2.0)
    for _ in range(24):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if _search_range_of(speed, m1, cfg) < _search_range_of(speed, m2, cfg):
            a = m1
        else:
            b = m2
    return (a + b) / 2.0


def max_range(
    speed: float,
    cfg: JetConfig,
    min_elevation_deg: float = 0.0,
    max_elevation_deg: float = 45.0,
) -> float:
    return range_of(
        speed, peak_elevation(speed, cfg, min_elevation_deg, max_elevation_deg), cfg
    )


def solve_elevation(
    target_range_m: float,
    speed: float,
    cfg: JetConfig,
    min_elevation_deg: float,
    max_elevation_deg: float,
) -> float | None:
    """Elevation whose landing distance equals the target.

    Range is unimodal in elevation (rises to a peak ~30° under drag, then
    falls), so a target below the peak range has up to two solutions. The
    **low arc** is preferred — flatter, shorter flight time, less wind
    exposure — and used whenever it can reach. For a target closer than the
    minimum-elevation range, the **high arc** (a lob past the peak) is used
    instead. Returns None when the target is unreachable at this speed.
    """
    peak = peak_elevation(speed, cfg, min_elevation_deg, max_elevation_deg)
    r_min = _search_range_of(speed, min_elevation_deg, cfg)
    r_peak = _search_range_of(speed, peak, cfg)
    r_max = _search_range_of(speed, max_elevation_deg, cfg)

    if target_range_m > r_peak:
        return None  # beyond this speed's reach at any elevation

    if target_range_m >= r_min and peak > min_elevation_deg:
        # low arc: range increases from min_elevation to peak
        lo, hi = min_elevation_deg, peak
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if _search_range_of(speed, mid, cfg) < target_range_m:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # target closer than the low arc can manage: high-arc lob if reachable
    if target_range_m >= r_max and max_elevation_deg > peak:
        lo, hi = peak, max_elevation_deg  # range decreases across this branch
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if _search_range_of(speed, mid, cfg) > target_range_m:
                lo = mid  # still too far, raise elevation to shorten
            else:
                hi = mid
        return (lo + hi) / 2.0

    return None


def solve_pump(
    target_range_m: float,
    elevation_deg: float,
    cfg: JetConfig,
    min_pump_pct: float | None = None,
) -> float | None:
    """Pump percent that lands water at `target_range_m` for a FIXED elevation,
    by bisection. Range is monotonically increasing in pump at fixed elevation,
    which makes pump the clean single-axis range control during suppression.
    Returns None if the target is unreachable within the pump range.
    """
    lo = cfg.min_pump_pct if min_pump_pct is None else min_pump_pct
    hi = 100.0
    if lo >= hi:
        return None
    r_lo = range_of(exit_velocity(lo, cfg), elevation_deg, cfg)
    r_hi = range_of(exit_velocity(hi, cfg), elevation_deg, cfg)
    if target_range_m < r_lo or target_range_m > r_hi:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if range_of(exit_velocity(mid, cfg), elevation_deg, cfg) < target_range_m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class FiringSolution:
    pump_pct: float
    elevation_deg: float
    predicted_range_m: float
    speed_mps: float


@dataclass(frozen=True)
class FitResult:
    velocity_coeff: float
    drag_k: float
    rmse_m: float


def fit_jet(shots: Sequence[tuple[float, float, float]], base_jet: JetConfig) -> FitResult:
    """Grid-search velocity_coeff and drag_k to minimise RMSE against measured
    shots ``(pump_pct, elevation_deg, measured_range_m)``. Same search as the
    ``fireturret fit`` CLI, reused by the Studio calibration panel."""
    if not shots:
        raise ValueError("no shots to fit")
    best: tuple[float, float, float] | None = None
    for cv in (x / 100.0 for x in range(60, 101, 2)):
        for k in (x / 1000.0 for x in range(20, 401, 10)):
            jet = replace(base_jet, velocity_coeff=cv, drag_k=k)
            err = 0.0
            for pump, elev, measured in shots:
                predicted = range_of(exit_velocity(pump, jet), elev, jet)
                err += (predicted - measured) ** 2
            if best is None or err < best[0]:
                best = (err, cv, k)
    assert best is not None  # grid is non-empty and shots was checked above
    total_err, cv, k = best
    return FitResult(cv, k, (total_err / len(shots)) ** 0.5)


def reach_bounds(
    cfg: JetConfig, min_elevation_deg: float, max_elevation_deg: float
) -> tuple[float, float]:
    """The jet's practical (min, max) reachable ground range: shortest coherent
    shot (min pump, min elevation) to maximum reach (full pump, peak elevation).
    """
    peak = peak_elevation(exit_velocity(100.0, cfg), cfg, min_elevation_deg, max_elevation_deg)
    max_reach = range_of(exit_velocity(100.0, cfg), peak, cfg)
    min_reach = range_of(exit_velocity(cfg.min_pump_pct, cfg), min_elevation_deg, cfg)
    return min_reach, max_reach


def plan_suppression(
    target_range_m: float,
    cfg: JetConfig,
    min_elevation_deg: float,
    max_elevation_deg: float,
) -> FiringSolution | None:
    """Choose a (tilt, pump) opening solution that puts the target in the
    MIDDLE of the pump's range envelope, so the closed loop can trim pump both
    up and down. Fixes tilt on the low arc; pump becomes the range control.

    Prefers the highest of a set of mid pump levels that can still reach the
    target (maximising downward headroom); escalates to higher pumps only for
    far targets.
    """
    candidates = [55.0, 45.0, 35.0, 25.0, cfg.min_pump_pct, 70.0, 85.0, 100.0]
    seen: set[float] = set()
    for pump in candidates:
        pump = max(cfg.min_pump_pct, min(100.0, pump))
        if pump in seen:
            continue
        seen.add(pump)
        speed = exit_velocity(pump, cfg)
        tilt = solve_elevation(
            target_range_m, speed, cfg, min_elevation_deg, max_elevation_deg
        )
        if tilt is not None:
            return FiringSolution(
                pump_pct=pump,
                elevation_deg=tilt,
                predicted_range_m=range_of(speed, tilt, cfg),
                speed_mps=speed,
            )
    return None


def choose_solution(
    target_range_m: float,
    cfg: JetConfig,
    min_elevation_deg: float,
    max_elevation_deg: float,
    headroom: float = 1.15,
) -> FiringSolution | None:
    """Pick the lowest pump setting whose max reach clears the target with
    headroom (so the servo can push further without saturating), then solve
    the elevation at that setting. None if even full pressure cannot reach.
    """
    start_pct = int(round(min(100.0, max(0.0, cfg.min_pump_pct))))
    for pct in range(start_pct, 101, 5):
        speed = exit_velocity(float(pct), cfg)
        if max_range(speed, cfg, min_elevation_deg, max_elevation_deg) < target_range_m * headroom:
            continue
        elev = solve_elevation(
            target_range_m, speed, cfg, min_elevation_deg, max_elevation_deg
        )
        if elev is None:
            continue
        return FiringSolution(
            pump_pct=float(pct),
            elevation_deg=elev,
            predicted_range_m=range_of(speed, elev, cfg),
            speed_mps=speed,
        )
    # last resort: full pressure even without headroom
    speed = exit_velocity(100.0, cfg)
    elev = solve_elevation(target_range_m, speed, cfg, min_elevation_deg, max_elevation_deg)
    if elev is not None:
        return FiringSolution(100.0, elev, range_of(speed, elev, cfg), speed)
    return None
