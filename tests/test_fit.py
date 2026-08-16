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


def test_fit_jet_refuses_a_single_operating_point():
    """Two parameters cannot be identified from one (pump, elevation) setting.

    Repeating the same shot satisfies the callers' "at least 3 shots" guard while
    leaving velocity_coeff and drag_k on a flat ridge: every pair that reproduces
    that one range fits it equally well, so the grid search returns whichever
    corner it happened to visit first, with an RMSE near zero to vouch for it.
    That is the failure the rest of this codebase names explicitly -- fitting
    unidentifiable parameters produces confident nonsense, not a weak estimate --
    so the fit has to decline rather than answer.
    """
    with pytest.raises(ValueError, match="operating point"):
        fit_jet([(50.0, 20.0, 7.0)] * 3, DEFAULT_CONFIG.jet)


def test_fit_jet_refuses_repeated_settings_however_many_shots():
    """Repeat measurements are good practice and still not identifiability: a
    dozen shots at two pump levels but one elevation each is two settings, and a
    dozen at one setting is still one."""
    with pytest.raises(ValueError, match="operating point"):
        fit_jet([(50.0, 20.0, 6.9), (50.0, 20.0, 7.0), (50.0, 20.0, 7.1)] * 4,
                DEFAULT_CONFIG.jet)
