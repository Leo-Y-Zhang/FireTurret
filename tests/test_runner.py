# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP3 — the per-tick deadline watchdog.

The failure it exists to catch: a control loop degraded from 30 Hz to 4 Hz keeps
the firmware's 500 ms heartbeat perfectly fed while servoing on quarter-second-old
observations with the valve open. The firmware watchdog catches a hung host, not
a slow one.

Everything here is clock-injected, so these assertions are about behaviour rather
than about how loaded the machine was when they ran.
"""

from __future__ import annotations

import pytest

from triton.rig.interface import RigCommand
from triton.runner import (
    DEFAULT_DEADLINE_S,
    TickWatchdog,
)

WET = RigCommand(pan_deg=10.0, tilt_deg=20.0, pump_pct=60.0, valve=True)
LATE = DEFAULT_DEADLINE_S * 2
FINE = DEFAULT_DEADLINE_S / 2


def _wd(**kw) -> TickWatchdog:
    return TickWatchdog(**kw)


def test_a_fast_tick_is_healthy() -> None:
    health = _wd().observe(FINE)
    assert health.healthy
    assert not health.overrun
    assert not health.shed_optional
    assert not health.water_cut


def test_one_late_tick_sheds_optional_work_but_keeps_water() -> None:
    """Dropping the overlay is cheap and reversible, and doing it at the first
    sign of trouble is often enough to recover. Cutting water is not cheap."""
    health = _wd().observe(LATE)
    assert health.overrun
    assert health.shed_optional
    assert health.water_cut is False


def test_a_single_expensive_tick_does_not_cut_water() -> None:
    """`plan_suppression` measures ~53 ms — 1.6 frames. Entering RANGE must not
    look like a failing loop."""
    wd = _wd()
    health = wd.observe(0.053)
    assert health.overrun is True
    assert health.water_cut is False


def test_sustained_overruns_cut_water() -> None:
    wd = _wd(max_consecutive_overruns=5)
    for _ in range(4):
        assert wd.observe(LATE).water_cut is False
    assert wd.observe(LATE).water_cut is True


def test_the_cut_forces_pump_and_valve_off_without_re_aiming() -> None:
    wd = _wd(max_consecutive_overruns=1)
    wd.observe(LATE)
    gated = wd.gate(WET)
    assert gated.valve is False
    assert gated.pump_pct == 0.0
    assert gated.pan_deg == WET.pan_deg  # deny-only: never re-aims
    assert gated.tilt_deg == WET.tilt_deg


def test_water_is_untouched_while_healthy() -> None:
    wd = _wd()
    wd.observe(FINE)
    assert wd.gate(WET) == WET


def test_one_good_tick_does_not_clear_a_cut() -> None:
    """A loop oscillating either side of its deadline is not healthy, and
    flapping the valve at its beat frequency is worse than holding it shut."""
    wd = _wd(max_consecutive_overruns=1, recovery_ticks=10)
    wd.observe(LATE)
    assert wd.water_cut is True
    assert wd.observe(FINE).water_cut is True
    assert wd.gate(WET).valve is False


def test_sustained_recovery_clears_the_cut() -> None:
    wd = _wd(max_consecutive_overruns=1, recovery_ticks=10)
    wd.observe(LATE)
    for _ in range(9):
        assert wd.observe(FINE).water_cut is True
    assert wd.observe(FINE).water_cut is False
    assert wd.gate(WET) == WET


def test_an_alternating_loop_never_recovers() -> None:
    """The scenario the recovery streak exists for: late, fine, late, fine...
    Each good tick resets nothing, because the loop is still missing half its
    deadlines."""
    wd = _wd(max_consecutive_overruns=2, recovery_ticks=5)
    wd.observe(LATE)
    wd.observe(LATE)
    assert wd.water_cut is True
    for _ in range(20):
        wd.observe(FINE)
        wd.observe(LATE)
    assert wd.water_cut is True, "an alternating loop was declared healthy"


def test_consecutive_counter_resets_on_a_good_tick() -> None:
    wd = _wd(max_consecutive_overruns=5)
    wd.observe(LATE)
    wd.observe(LATE)
    assert wd.observe(FINE).consecutive_overruns == 0


# ------------------------------------------------------------- clock plumbing

def test_begin_end_measures_with_the_injected_clock() -> None:
    ticks = iter([100.0, 100.0 + LATE])
    wd = _wd(clock=lambda: next(ticks))
    wd.begin()
    health = wd.end()
    assert health.duration_s == pytest.approx(LATE)
    assert health.overrun is True


def test_end_without_begin_is_an_error_not_a_silent_zero() -> None:
    """A zero-duration tick would read as perfectly healthy, which is the most
    dangerous possible answer to give by accident."""
    with pytest.raises(RuntimeError, match="begin"):
        _wd().end()


@pytest.mark.parametrize("kwargs", [
    {"deadline_s": 0.0},
    {"deadline_s": -1.0},
    {"max_consecutive_overruns": 0},
])
def test_degenerate_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        TickWatchdog(**kwargs)
