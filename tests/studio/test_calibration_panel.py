# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P4: the jet calibration panel (fit measured shots, apply to config)."""
from __future__ import annotations

from dataclasses import replace

from triton.ballistics import exit_velocity, range_of
from triton.studio.session import Session
from triton.studio.views.calibration_panel import CalibrationPanel


def _shots_from(jet):
    return [
        (p, e, range_of(exit_velocity(p, jet), e, jet))
        for p, e in [(50, 20), (70, 15), (90, 25), (60, 30)]
    ]


def test_fit_and_apply(qapp):
    s = Session()
    # target differs from the defaults (0.90/0.16) so applying actually changes config
    true = replace(s.config.jet, velocity_coeff=0.84, drag_k=0.20)
    panel = CalibrationPanel(s)
    panel.set_shots(_shots_from(true))
    result = panel.fit()
    assert result is not None
    assert abs(result.velocity_coeff - 0.84) < 0.05
    panel.apply_fit()
    assert abs(s.config.jet.velocity_coeff - 0.84) < 0.05
    assert abs(s.config.jet.drag_k - 0.20) < 0.03
    assert s.dirty is True


def test_needs_three_shots(qapp):
    panel = CalibrationPanel(Session())
    panel.set_shots([(50, 20, 4.5)])
    assert panel.fit() is None
