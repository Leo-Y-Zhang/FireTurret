# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P2: the read-only top-down scene view."""
from __future__ import annotations

import numpy as np

from fireturret.ballistics import reach_bounds
from fireturret.config import DEFAULT_CONFIG
from fireturret.rig.sim_rig import SimScenario
from fireturret.studio.views.topdown import TopDownView


def test_fire_placed_straight_ahead(qapp):
    w = TopDownView()
    w.set_scene(SimScenario(fire_azimuth_deg=0.0, fire_range_m=7.0),
                DEFAULT_CONFIG.jet, (8.0, 50.0))
    xs, ys = w.fires.getData()
    assert abs(xs[0]) < 1e-6      # az=0 -> x≈0
    assert abs(ys[0] - 7.0) < 1e-6  # forward≈range


def test_extra_fires_and_decoy(qapp):
    w = TopDownView()
    w.set_scene(SimScenario(extra_fires=((30.0, 6.0),), decoy=(-30.0, 5.0)),
                DEFAULT_CONFIG.jet, (8.0, 50.0))
    assert len(w.fires.getData()[0]) == 2  # primary + one extra
    assert len(w.decoy.getData()[0]) == 1


def test_no_decoy_leaves_empty(qapp):
    w = TopDownView()
    w.set_scene(SimScenario(), DEFAULT_CONFIG.jet, (8.0, 50.0))
    assert len(w.decoy.getData()[0]) == 0


def test_max_ring_radius_matches_reach(qapp):
    w = TopDownView()
    w.set_scene(SimScenario(), DEFAULT_CONFIG.jet, (8.0, 50.0))
    _lo, hi = reach_bounds(DEFAULT_CONFIG.jet, 8.0, 50.0)
    xs, ys = w._ring_max.getData()
    radius = np.sqrt(xs**2 + ys**2)
    assert abs(radius.max() - hi) < 1e-6
