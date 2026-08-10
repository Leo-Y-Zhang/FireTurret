# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Jet calibration fit (shared by the CLI and the Studio calibration panel)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fireturret.ballistics import exit_velocity, fit_jet, range_of
from fireturret.config import DEFAULT_CONFIG


def test_fit_jet_recovers_known_params():
    true = replace(DEFAULT_CONFIG.jet, velocity_coeff=0.90, drag_k=0.16)
    shots = [
        (p, e, range_of(exit_velocity(p, true), e, true))
        for p, e in [(50, 20), (70, 15), (90, 25), (60, 30)]
    ]
    result = fit_jet(shots, DEFAULT_CONFIG.jet)
    assert abs(result.velocity_coeff - 0.90) < 0.05
    assert abs(result.drag_k - 0.16) < 0.03
    assert result.rmse_m < 0.5


def test_fit_jet_empty_shots_raises_valueerror():
    """No shots to fit must raise the intended ValueError, not a confusing
    ZeroDivisionError from the rmse average (0.0 / 0)."""
    with pytest.raises(ValueError, match="no shots to fit"):
        fit_jet([], DEFAULT_CONFIG.jet)
