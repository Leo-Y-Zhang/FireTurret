# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P4: the advisories panel surfaces the operator warnings a run raised."""
from __future__ import annotations

from fireturret.__main__ import SCENARIOS
from fireturret.config import DEFAULT_CONFIG
from fireturret.simcore import drive
from fireturret.studio.views.advisories_view import AdvisoriesView


def test_lists_raised_advisories(qapp):
    # the `unreachable` scenario provably raises advisories (fire out of reach)
    report = drive(DEFAULT_CONFIG, SCENARIOS["unreachable"], seed=7,
                   max_frames=400, record_telemetry=True)
    view = AdvisoriesView()
    view.set_report(report)
    assert report.advisories_raised
    assert view.kinds == sorted(report.advisories_raised)
    assert view._list.count() == len(report.advisories_raised)


def test_empty_state(qapp):
    view = AdvisoriesView()
    view.set_report(None)
    assert view.kinds == []
    assert view._list.count() == 1  # a single "no advisories" placeholder row
