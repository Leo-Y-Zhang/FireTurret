# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Real-time tick supervision.

## The gap this closes

The firmware's heartbeat watchdog cuts water if no valid command arrives for
500 ms. That catches a *hung* host. It does not catch a host that is still
sending commands, just slowly — a control loop degraded from 30 Hz to 4 Hz keeps
the heartbeat perfectly fed while the visual servo is running on quarter-second-old
observations with the valve open. **Nothing on the host currently notices that.**

It is not a hypothetical: `ballistics.plan_suppression` measures at ~53 ms, which
is 1.6 frames at 30 fps. Today that is survivable because RANGE is entered once
per engagement. From SP8 onward the target's range changes continuously.

## The response, in two stages

1. **Shed optional work.** Overlay rendering, recording and telemetry export are
   luxuries; perception and control are not. On overrun the optional subsystems
   are dropped first, which is often enough to recover.
2. **Cut water.** After `max_consecutive_overruns` ticks that are still late, the
   loop is declared unfit to aim and water is forced off. Recovery requires
   sustained healthy ticks, not a single lucky one — a loop oscillating either
   side of its deadline is not healthy, and flapping the valve is worse than
   holding it shut.

## Why this is off in the simulator

The watchdog measures WALL-CLOCK time against a real-time deadline. The simulator
advances a fixed `dt = 1/30` per tick as fast as the machine allows: its ticks
have no real-time meaning, it is not driving hardware, and there is no water to
cut. Enabling it there would make golden fixtures a function of how loaded the
build machine was — which is precisely the class of coupling SP2 removed. So it
is opt-in.

## NOT WIRED YET — read this before relying on it

This used to end "and the real-time paths (`fireturret run`, `fireturret web`) opt in".
They do not. `TickWatchdog` is referenced nowhere in `src/` outside this file:
`app.run_capture` and `ui.webserver.serve` contain no timing measurement, never
call `begin()`/`end()`/`observe()`, and never pass a command through `gate()`.
The only importer in the repository is `tests/test_runner.py`, which exercises
the class in isolation, so no test notices the omission either.

The consequence is exactly the failure this module was written to close. A loop
degraded from 30 Hz to 4 Hz keeps calling `rig.command()` every iteration, so the
firmware's 500 ms heartbeat stays fed and the board never cuts water, while the
servo acts on splash observations a quarter-second stale. `shed_optional` never
fires and `water_cut` never becomes True, because nothing is measuring.

Wiring it is left to a human rather than done blind: the watchdog is disabled in
the simulator BY DESIGN (above), so the sim suite structurally cannot exercise a
change to the hardware paths, and an untested edit to the real-time control loop
of a machine that sprays water is worse than an honest gap. To close it:
construct a `TickWatchdog` in `run_capture` and in `serve`'s non-sim branch,
bracket the tick with `begin()`/`end()`, skip overlay and recording when
`health.shed_optional`, pass the command through `watchdog.gate(...)` before
`rig.command(...)`, and add a wiring assertion mirroring the one in
`tests/test_safety_debt.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from .rig.interface import RigCommand

# One tick at 30 fps. A tick that takes longer than this has, by definition,
# missed its slot.
DEFAULT_DEADLINE_S = 1.0 / 30.0

# Consecutive late ticks tolerated before water is cut. Sized so a single
# expensive tick — entering RANGE and planning a firing solution, ~53 ms — cannot
# trip it, while a loop that is genuinely running slow does so in ~1/6 second.
DEFAULT_MAX_OVERRUNS = 5

# Consecutive healthy ticks required to clear a water cut. Greater than one so a
# loop oscillating either side of its deadline stays cut instead of flapping the
# valve at its own beat frequency.
DEFAULT_RECOVERY_TICKS = 10


@dataclass(frozen=True)
class TickHealth:
    """What the watchdog concluded about the tick just measured."""

    duration_s: float
    overrun: bool
    consecutive_overruns: int
    shed_optional: bool  # skip overlay/record/export work on the next tick
    water_cut: bool      # the loop is too slow to be trusted with water

    @property
    def healthy(self) -> bool:
        return not self.overrun and not self.water_cut


class TickWatchdog:
    """Measures tick duration and decides whether the loop is fit to aim.

    Pure and clock-injected: `now` is a parameter so tests are deterministic
    rather than timing-dependent, which is the difference between a test that
    documents behaviour and one that fails on a busy CI box.
    """

    def __init__(
        self,
        deadline_s: float = DEFAULT_DEADLINE_S,
        max_consecutive_overruns: int = DEFAULT_MAX_OVERRUNS,
        recovery_ticks: int = DEFAULT_RECOVERY_TICKS,
        clock=time.perf_counter,
    ) -> None:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if max_consecutive_overruns < 1:
            raise ValueError("max_consecutive_overruns must be at least 1")
        self.deadline_s = deadline_s
        self.max_consecutive_overruns = max_consecutive_overruns
        self.recovery_ticks = recovery_ticks
        self._clock = clock
        self._started: float | None = None
        self.consecutive_overruns = 0
        self.healthy_streak = 0
        self.water_cut = False

    def begin(self) -> None:
        self._started = self._clock()

    def end(self) -> TickHealth:
        """Close the tick opened by `begin` and judge it."""
        if self._started is None:
            raise RuntimeError("TickWatchdog.end() called without begin()")
        duration = self._clock() - self._started
        self._started = None
        return self.observe(duration)

    def observe(self, duration_s: float) -> TickHealth:
        """Judge a tick of known duration. Separated from `end` so a caller that
        already measures its own timing does not have to fake a clock."""
        overrun = duration_s > self.deadline_s
        if overrun:
            self.consecutive_overruns += 1
            self.healthy_streak = 0
            if self.consecutive_overruns >= self.max_consecutive_overruns:
                self.water_cut = True
        else:
            self.consecutive_overruns = 0
            self.healthy_streak += 1
            if self.water_cut and self.healthy_streak >= self.recovery_ticks:
                self.water_cut = False
                self.healthy_streak = 0

        return TickHealth(
            duration_s=duration_s,
            overrun=overrun,
            consecutive_overruns=self.consecutive_overruns,
            # shed as soon as ONE tick is late: dropping the overlay is cheap and
            # reversible, and doing it early is often enough to avoid the cut
            shed_optional=overrun or self.water_cut,
            water_cut=self.water_cut,
        )

    def gate(self, cmd: RigCommand) -> RigCommand:
        """Force water off while the loop is unfit. Deny-only, like the water
        gate: it never re-aims, and it can never turn water back on."""
        if not self.water_cut:
            return cmd
        return replace(cmd, pump_pct=0.0, valve=False)
