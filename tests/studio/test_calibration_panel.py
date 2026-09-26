# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P4: the jet calibration panel (fit measured shots, apply to config)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fireturret.ballistics import exit_velocity, range_of
from fireturret.studio.session import Session
from fireturret.studio.views.calibration_panel import CalibrationPanel


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


@pytest.mark.parametrize("refused", [
    [(50, 20, 4.5)],                # too few shots
    [(50.0, 20.0, 7.0)] * 3,        # one operating point, which fit_jet refuses
])
def test_a_refused_refit_leaves_no_earlier_fit_to_apply(qapp, refused):
    """Apply writes the last fit into the model. Once a refit is refused, the
    earlier fit belongs to shots that are no longer in the table and the label no
    longer shows it, so Apply must not be able to write it."""
    s = Session()
    panel = CalibrationPanel(s)
    panel.set_shots(_shots_from(replace(s.config.jet, velocity_coeff=0.84, drag_k=0.20)))
    assert panel.fit() is not None
    panel.set_shots(refused)
    assert panel.fit() is None
    panel.apply_fit()
    assert s.config.jet == Session().config.jet
    assert s.dirty is False
