# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
from fireturret.control.advisories import (
    CRITICAL,
    WARN,
    Advisory,
    most_urgent,
    reachability_advisory,
    traverse_advisory,
)


def test_too_far_advises_moving_closer() -> None:
    a = reachability_advisory(12.0, 4.5, 11.0)
    assert a is not None
    assert a.kind == "TOO_FAR"
    assert "closer" in a.action.lower()


def test_too_close_advises_moving_back() -> None:
    a = reachability_advisory(3.0, 4.5, 11.0)
    assert a is not None
    assert a.kind == "TOO_CLOSE"
    assert "back" in a.action.lower()


def test_reachable_range_has_no_advisory() -> None:
    assert reachability_advisory(7.0, 4.5, 11.0) is None


def test_reachability_severity_is_configurable() -> None:
    assert reachability_advisory(20.0, 4.5, 11.0, severity=CRITICAL).severity == CRITICAL


def test_traverse_advises_rotating_base() -> None:
    right = traverse_advisory(168.0, -170.0, 170.0)
    assert right is not None and "right" in right.action.lower()
    left = traverse_advisory(-168.0, -170.0, 170.0)
    assert left is not None and "left" in left.action.lower()
    assert traverse_advisory(0.0, -170.0, 170.0) is None


def test_most_urgent_picks_highest_severity() -> None:
    low = Advisory("OUT_OF_TRAVERSE", WARN, "m", "a")
    high = Advisory("SAFETY", CRITICAL, "m", "a")
    assert most_urgent([low, high]) is high
    assert most_urgent([]) is None
