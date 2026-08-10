# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 9: the ballistics explorer redraws the arc + envelope from the jet model."""
from __future__ import annotations

import pytest

from fireturret.ballistics import exit_velocity, reach_bounds, simulate_arc
from fireturret.config import DEFAULT_CONFIG
from fireturret.studio.views.ballistics_view import BallisticsView


def test_set_elevation_updates_arc(qapp):
    jet = DEFAULT_CONFIG.jet
    w = BallisticsView()
    w.set_jet(jet, (8.0, 50.0))
    w.set_pump(60)
    w.set_elevation(30)

    xs, ys = w.arc_curve.getData()
    expected = simulate_arc(exit_velocity(60.0, jet), 30.0, jet)
    assert list(xs) == [p[0] for p in expected]
    assert list(ys) == [p[1] for p in expected]


def test_pump_change_lengthens_range(qapp):
    jet = DEFAULT_CONFIG.jet
    w = BallisticsView()
    w.set_jet(jet, (8.0, 50.0))
    w.set_elevation(22)
    w.set_pump(40)
    short = w.arc_curve.getData()[0][-1]
    w.set_pump(90)
    long = w.arc_curve.getData()[0][-1]
    assert long > short


def test_true_arc_shorter_than_model_when_perturbed(qapp):
    jet = DEFAULT_CONFIG.jet
    w = BallisticsView()
    w.set_jet(jet, (8.0, 50.0))
    w.set_pump(80)
    w.set_elevation(25)
    w.set_true_scales(0.88, 1.3)  # weaker jet + more drag -> shorter true range
    model_range = w.arc_curve.getData()[0][-1]
    true_range = w.true_curve.getData()[0][-1]
    assert 0 < true_range < model_range


def test_set_jet_resyncs_elevation_when_bounds_clamp_slider(qapp):
    """Editing the tilt bounds to exclude the current elevation must re-sync the
    cached elevation, label, and drawn arc to the clamped slider value — not leave
    the arc drawn at a stale, now-out-of-bounds elevation."""
    jet = DEFAULT_CONFIG.jet
    w = BallisticsView()
    w.set_jet(jet, (8.0, 50.0))  # default elevation 22 is in range here
    w.set_jet(jet, (25.0, 50.0))  # new min 25 excludes 22 -> slider clamps to 25
    assert w._elevation == pytest.approx(float(w._elev_slider.value()))
    assert w._elevation == pytest.approx(25.0)
    assert w._elev_label.text() == "25"
    # the drawn arc uses the clamped, in-bounds elevation
    xs, _ = w.arc_curve.getData()
    expected = simulate_arc(exit_velocity(w._pump, jet), 25.0, jet)
    assert list(xs) == [p[0] for p in expected]


def test_envelope_matches_reach_bounds(qapp):
    jet = DEFAULT_CONFIG.jet
    w = BallisticsView()
    w.set_jet(jet, (8.0, 50.0))
    lo, hi = reach_bounds(jet, 8.0, 50.0)
    region = w.envelope.getRegion()
    assert region[0] == pytest.approx(lo)
    assert region[1] == pytest.approx(hi)
