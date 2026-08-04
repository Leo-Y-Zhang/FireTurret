# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1b: the tree + properties model editor, bound to the Session."""
from __future__ import annotations

import pytest

from triton.studio.session import Session
from triton.studio.views.model_editor import ModelEditor


def test_editing_a_jet_field_updates_session(qapp):
    s = Session()
    ed = ModelEditor(s)
    ed.select_group("jet")
    ed._widgets[("jet", "drag_k")].setValue(0.3)
    assert s.config.jet.drag_k == pytest.approx(0.3)
    assert s.dirty is True


def test_building_panel_does_not_dirty_session(qapp):
    s = Session()
    ModelEditor(s)  # lands on the "jet" group, populating widgets
    assert s.dirty is False  # initial population must not commit edits


def test_scenario_group_edit(qapp):
    s = Session()
    ed = ModelEditor(s)
    ed.select_group("scenario")
    ed._widgets[("scenario", "fire_range_m")].setValue(8.5)
    assert s.scenario.fire_range_m == pytest.approx(8.5)


def test_widgets_refresh_on_undo(qapp):
    s = Session()
    ed = ModelEditor(s)
    ed.select_group("jet")
    orig = s.config.jet.drag_k
    s.set_field("jet", "drag_k", 0.3)
    s.undo()
    assert ed._widgets[("jet", "drag_k")].value() == pytest.approx(orig)


def test_all_groups_have_widgets(qapp):
    s = Session()
    ed = ModelEditor(s)
    for key in ("camera", "turret", "jet", "detector", "servo", "mission", "scenario"):
        ed.select_group(key)
        assert ed._widgets, f"no widgets built for group {key}"
