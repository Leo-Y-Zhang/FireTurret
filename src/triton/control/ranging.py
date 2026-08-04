# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Where the target is, and how expensive it is to ask.

Two things live here, lifted out of `MissionController` so the suppression law
cannot reach for them:

1. **`target_ground_px` + `RangeEstimator`** — the monocular ground-plane range
   estimate and its low-pass filter. This is the number the architecture
   deliberately does NOT servo on (invariant 1: the suppression loop drives the
   splash pixel onto the fire pixel, in image space). It is used to pick an
   OPENING solution and to drive operator advisories, and that is all. Keeping it
   in its own module is what makes that boundary checkable — `control/suppress.py`
   is AST-tested against importing any of it.

2. **`SpeedRangeTable` / `RangeTable`** — bilinear interpolants over
   `ballistics.range_of`, built once per `JetConfig`.

## Why the tables exist

Measured on an idle machine: `ballistics.plan_suppression` cost **67.3 ms** — two
frames at 30 fps — because it integrates a fresh RK4 arc at every point of an
elevation search, and the search calls `range_of` hundreds of times. (It measured
43.7 ms when first characterised; the figure moves with machine state, and both
are over budget, which is the part that matters.) That was invisible only because
RANGE is entered once per engagement; on a moving platform, where the target's
range changes continuously and a solution is needed every cycle, it is a missed
deadline rather than a detail.

`SpeedRangeTable` is now wired into `ballistics`, which took the same call to
**0.606 ms — a 111x speedup**, comfortably inside budget. Building the table
costs ~500 ms once per `JetConfig`, cached thereafter. `RangeTable` (indexed by
pump percentage) is retained for callers that hold a pump setting rather than a
speed; nothing in the shipped loop uses it.

## What adopting it cost, precisely

Interpolation is an approximation, so it had to be measured before it was
believed. Replayed over the 120 frozen tier-3 frames against the byte-exact
golden commands:

    pan_deg    max |delta|  0.000000
    pump_pct   max |delta|  0.000000
    tilt_deg   max |delta|  0.036093   (112 of 120 frames)
    valve / laser / warn decisions differing   0

Every safety-relevant output is unchanged. The whole residual is 0.036 deg of
tilt — **7.5 mm of range at 7.2 m**, and **1.8% of the tilt the turret slews in
one 33 ms tick**. The table is used only inside the elevation SEARCH; the
`predicted_range_m` that leaves `plan_suppression` is still computed by exact
integration at the chosen elevation, so nothing downstream inherits
interpolation error.

A query outside the grid does NOT interpolate — `covers()` sends it to exact
integration. Without that, asking for a high-arc solution at 80 deg would have
quietly received the range at 60 deg: the characteristic lookup-table bug, which
is not imprecision but a confident answer to a question the table was never
built for.

This is the fourth golden regeneration, and the argument for it is written up in
the fixture's recorded reason in `tests/golden/MANIFEST.json` rather than
assumed.
"""

from __future__ import annotations

import numpy as np

from ..ballistics import exit_velocity, range_of
from ..config import JetConfig, MissionConfig
from ..geometry import CameraModel, clamp
from ..vision.tracker import Track

# The EMA's weights and the sanity clamp, verbatim from MissionController. The
# filter exists because the bbox bottom flickers with the flame, and an unfiltered
# range estimate would jitter the opening solution frame to frame.
_EMA_KEEP = 0.7
_EMA_NEW = 0.3
_RANGE_MIN_M = 1.5
_RANGE_MAX_M = 40.0


def target_ground_px(target: Track) -> tuple[float, float]:
    """Aim at the base of the flames — where the burning material meets the
    ground — not the flame centroid, which sits above it and would bias every
    range estimate long."""
    if target.last_blob is not None:
        _x, y, _w, h = target.last_blob.bbox
        return (target.cx, float(y + h))
    return (target.cx, target.cy)


class RangeEstimator:
    """Low-pass filtered ground-plane range to the flame base.

    Holds the EMA that used to be `MissionController._target_range_ema`. Extracted
    unchanged: same weights, same clamp, same fallback order, so the opening
    solution it feeds is bit-identical.
    """

    def __init__(self, camera: CameraModel, mission: MissionConfig) -> None:
        self._camera = camera
        self._mission = mission
        self.ema: float | None = None

    def reset(self) -> None:
        self.ema = None

    def update(self, target: Track) -> float:
        est = self._camera.ground_range_from_px(*target_ground_px(target))
        if est is None:
            # No ground-plane intersection (the flame base is above the horizon,
            # or the geometry is degenerate). Fall back to the last good estimate,
            # then to the configured bootstrap.
            est = self.ema or self._mission.default_range_m
        est = clamp(est, _RANGE_MIN_M, _RANGE_MAX_M)
        if self.ema is None:
            self.ema = est
        else:
            self.ema = _EMA_KEEP * self.ema + _EMA_NEW * est
        return self.ema


class SpeedRangeTable:
    """A `RangeTable` indexed by exit SPEED rather than pump percentage.

    The solvers in `ballistics` work in speed — `solve_elevation(target, speed,
    ...)` — so a pump-indexed table would need an inverse of `exit_velocity` at
    every lookup, which is both slower and a second place for the pump-to-speed
    relationship to be written down.

    Elevation spans 0-90 deg rather than the turret's configured tilt limits,
    because `peak_elevation` searches the *physical* peak (~30 deg for a water
    jet) and a table clipped to the mount's limits would silently move it.
    """

    #: Upper tilt bound of the table. 60 deg rather than 90: the searches that
    #: use it look for the physical range peak (~30 deg for a water jet) inside
    #: the turret's own limits (8-50 deg by default), so 60 covers everything
    #: reachable with margin. The 30 deg of table above that would be 1/3 of the
    #: build cost spent on elevations nothing asks about.
    TILT_MAX_DEG = 60.0

    def __init__(self, jet: JetConfig, speed_steps: int = 25, tilt_steps: int = 25) -> None:
        from ..ballistics import exit_velocity, range_of

        self.jet = jet
        self.s0 = exit_velocity(jet.min_pump_pct, jet)
        self.s1 = exit_velocity(100.0, jet)
        self.t0, self.t1 = 0.0, self.TILT_MAX_DEG
        self.ns, self.nt = speed_steps, tilt_steps
        self.ds = (self.s1 - self.s0) / (speed_steps - 1)
        self.dt = (self.t1 - self.t0) / (tilt_steps - 1)

        # Plain nested lists of floats, not an ndarray. This is called several
        # hundred times per firing solution, and at that granularity numpy's
        # per-call overhead — boxing scalars, `searchsorted` on a 33-element
        # array — dominates the arithmetic completely. Measured: swapping numpy
        # indexing for the closed form below took plan_suppression from 17 ms to
        # under 1 ms, on the same grid.
        self.grid = [
            [range_of(self.s0 + i * self.ds, self.t0 + j * self.dt, jet)
             for j in range(tilt_steps)]
            for i in range(speed_steps)
        ]

    def covers(self, speed: float, elevation_deg: float) -> bool:
        """Whether this query is INSIDE the grid.

        Exists so callers can integrate exactly rather than silently accept an
        edge-clamped answer. `range_of_speed` deliberately still clamps — an
        interpolant that quietly extrapolated would be worse — but clamping is
        only the right answer when the caller has already decided it is.
        """
        return (self.s0 <= speed <= self.s1) and (self.t0 <= elevation_deg <= self.t1)

    def range_of_speed(self, speed: float, elevation_deg: float) -> float:
        """Bilinear lookup. The grids are UNIFORM, so the cell index is
        arithmetic rather than a search."""
        s = self.s0 if speed < self.s0 else (self.s1 if speed > self.s1 else speed)
        t = self.t0 if elevation_deg < self.t0 else (
            self.t1 if elevation_deg > self.t1 else elevation_deg
        )
        fi = (s - self.s0) / self.ds
        fj = (t - self.t0) / self.dt
        i = int(fi)
        j = int(fj)
        if i >= self.ns - 1:
            i = self.ns - 2
        if j >= self.nt - 1:
            j = self.nt - 2
        u = fi - i
        v = fj - j
        g0 = self.grid[i]
        g1 = self.grid[i + 1]
        return (
            g0[j] * (1.0 - u) * (1.0 - v)
            + g1[j] * u * (1.0 - v)
            + g0[j + 1] * (1.0 - u) * v
            + g1[j + 1] * u * v
        )


class RangeTable:
    """Bilinear interpolant of `range_of` over a (pump%, tilt deg) grid.

    Built once per `JetConfig`; after that a lookup is array arithmetic rather
    than an RK4 integration. See the module docstring for why that matters from
    SP8 onward, and why nothing consumes it yet.

    The grid is uniform in both axes because `range_of` is smooth and monotone in
    pump across the usable band — there is no knee to resolve, so a denser grid
    buys accuracy linearly and costs build time linearly.
    """

    def __init__(
        self,
        jet: JetConfig,
        tilt_min_deg: float,
        tilt_max_deg: float,
        pump_steps: int = 25,
        tilt_steps: int = 25,
    ) -> None:
        if pump_steps < 2 or tilt_steps < 2:
            raise ValueError("a bilinear grid needs at least 2 points per axis")
        self.jet = jet
        self.pumps = np.linspace(jet.min_pump_pct, 100.0, pump_steps)
        self.tilts = np.linspace(tilt_min_deg, tilt_max_deg, tilt_steps)
        self.grid = np.empty((pump_steps, tilt_steps), dtype=float)
        for i, pump in enumerate(self.pumps):
            speed = exit_velocity(float(pump), jet)
            for j, tilt in enumerate(self.tilts):
                self.grid[i, j] = range_of(speed, float(tilt), jet)

    def range_of(self, pump_pct: float, tilt_deg: float) -> float:
        """Interpolated range. Inputs outside the grid are clamped to its edge —
        the caller has already clamped pump and tilt to the actuator limits, so an
        out-of-grid query means a configuration change the table was not built
        for, and the edge value is the honest answer rather than an extrapolation.
        """
        pump = float(np.clip(pump_pct, self.pumps[0], self.pumps[-1]))
        tilt = float(np.clip(tilt_deg, self.tilts[0], self.tilts[-1]))

        i = int(np.searchsorted(self.pumps, pump) - 1)
        j = int(np.searchsorted(self.tilts, tilt) - 1)
        i = min(max(i, 0), len(self.pumps) - 2)
        j = min(max(j, 0), len(self.tilts) - 2)

        p0, p1 = self.pumps[i], self.pumps[i + 1]
        t0, t1 = self.tilts[j], self.tilts[j + 1]
        u = 0.0 if p1 == p0 else (pump - p0) / (p1 - p0)
        v = 0.0 if t1 == t0 else (tilt - t0) / (t1 - t0)

        g = self.grid
        return float(
            g[i, j] * (1 - u) * (1 - v)
            + g[i + 1, j] * u * (1 - v)
            + g[i, j + 1] * (1 - u) * v
            + g[i + 1, j + 1] * u * v
        )
