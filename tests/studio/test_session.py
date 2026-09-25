# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 6: the Studio Session document (validated edits + undo/redo)."""
from __future__ import annotations

import pytest

from fireturret.config import DEFAULT_CONFIG, config_from_dict, config_to_dict
from fireturret.studio.session import Session


def test_set_field_updates_config_and_sets_dirty(qapp):
    s = Session()
    assert s.dirty is False
    s.set_field("jet", "max_pressure_psi", 80.0)
    assert s.config.jet.max_pressure_psi == 80.0
    assert s.dirty is True


def test_invalid_value_raises_and_does_not_commit(qapp):
    s = Session()
    before = s.config.jet.max_pressure_psi
    with pytest.raises(ValueError):
        s.set_field("jet", "max_pressure_psi", -1.0)
    assert s.config.jet.max_pressure_psi == before
    assert s.dirty is False


def test_undo_redo_roundtrip(qapp):
    s = Session()
    orig = s.config.servo.pan_gain
    s.set_field("servo", "pan_gain", 0.9)
    assert s.config.servo.pan_gain == 0.9
    s.undo()
    assert s.config.servo.pan_gain == orig
    s.redo()
    assert s.config.servo.pan_gain == 0.9


def test_an_edit_keeps_plugin_config_groups(qapp):
    """A session opened with a plugin's config group must still carry it after an
    edit (and after undoing it), or saving the session destroys those settings."""
    d = config_to_dict(DEFAULT_CONFIG)
    d["some_plugin"] = {"threshold": 0.25}
    s = Session(config_from_dict(d))
    s.set_field("jet", "drag_k", 0.2)
    assert config_to_dict(s.config)["some_plugin"] == {"threshold": 0.25}
    s.undo()
    assert config_to_dict(s.config)["some_plugin"] == {"threshold": 0.25}


def test_scenario_field_edit(qapp):
    s = Session()
    s.set_field("scenario", "fire_range_m", 8.0)
    assert s.scenario.fire_range_m == 8.0


def test_unknown_group_raises(qapp):
    s = Session()
    with pytest.raises(ValueError):
        s.set_field("bogus", "x", 1.0)


def test_changed_signal_emitted(qapp, qtbot):
    s = Session()
    with qtbot.waitSignal(s.changed, timeout=1000):
        s.set_field("jet", "drag_k", 0.2)
