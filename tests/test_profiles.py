# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP12 — mission profiles and the plugin API.

The acceptance criterion the spec sets is identity: the fire profile must
reconstruct today's wiring object-for-object, so a run driven by it is
bit-identical to a run without one. An abstraction whose flagship case is not
provably identical to the code it replaced has not been extracted — it has been
re-implemented, and the difference surfaces later as a behaviour change nobody
meant to make.

The other half is the plugin API's failure modes, which are where the real risk
is: discovery that imports, an entry point that calls `sys.exit()` at module
scope, and a plugin that tries to take a built-in's name.
"""

from __future__ import annotations

import pytest

from fireturret.config import DEFAULT_CONFIG
from fireturret.geometry import CameraModel
from fireturret.plugins import (
    GROUP_DETECTOR,
    GROUP_PROFILE,
    GROUP_RIG,
    GROUPS,
    PluginError,
    builtin_names,
    discover,
    load,
)
from fireturret.profiles import FireProfile, MissionProfile, SafetyConstraints, WashdownProfile
from fireturret.profiles.washdown import SweepEngagement
from fireturret.testing import (
    ConformanceFailure,
    check_impact_observer,
    check_profile,
    check_rig,
)

CAM = CameraModel(DEFAULT_CONFIG.camera)


# ------------------------------------------------------- narrowing, only ever

def test_intersect_takes_the_stricter_ceiling() -> None:
    a = SafetyConstraints(max_spray_s=45.0, max_pump_pct=100.0)
    b = SafetyConstraints(max_spray_s=10.0, max_pump_pct=60.0)
    merged = a.intersect(b)
    assert merged.max_spray_s == 10.0
    assert merged.max_pump_pct == 60.0


def test_none_means_no_opinion_not_no_limit() -> None:
    a = SafetyConstraints(max_spray_s=45.0)
    assert a.intersect(SafetyConstraints()).max_spray_s == 45.0
    assert SafetyConstraints().intersect(a).max_spray_s == 45.0


def test_keepout_sectors_UNION_rather_than_intersect() -> None:
    """The asymmetry with the numeric fields is deliberate. A keep-out is a
    region water must NOT enter, so stricter means LARGER. Intersecting would
    let a second profile shrink a protected zone — exactly the widening this
    class exists to prevent."""
    a = SafetyConstraints(pan_keepout_deg=(-10.0, 10.0))
    b = SafetyConstraints(pan_keepout_deg=(5.0, 25.0))
    assert a.intersect(b).pan_keepout_deg == (-10.0, 25.0)


def test_require_armed_can_only_become_stricter() -> None:
    lax = SafetyConstraints(require_armed=False)
    strict = SafetyConstraints(require_armed=True)
    assert lax.intersect(strict).require_armed is True
    assert strict.intersect(lax).require_armed is True


@pytest.mark.parametrize("field,loose,tight", [
    ("max_spray_s", 45.0, 5.0),
    ("max_pump_pct", 100.0, 30.0),
])
def test_no_intersection_can_widen_a_limit(field, loose, tight) -> None:
    a = SafetyConstraints(**{field: loose})
    b = SafetyConstraints(**{field: tight})
    for merged in (a.intersect(b), b.intersect(a)):
        assert getattr(merged, field) == tight


def test_with_constraints_narrows_a_profile() -> None:
    profile = FireProfile().with_constraints(SafetyConstraints(max_pump_pct=40.0))
    assert profile.constraints.max_pump_pct == 40.0


# ------------------------------------------------------------- the fire profile

def test_the_fire_profile_aims_at_the_target() -> None:
    """Fire suppression is the case where the image setpoint offset is zero."""
    assert FireProfile().policy.setpoint_offset_px(None) == (0.0, 0.0)


def test_a_zero_setpoint_offset_leaves_the_servo_unchanged() -> None:
    """The identity property, at the level it actually matters: the generalised
    servo with a zero offset must produce exactly the old numbers."""
    from fireturret.control.servo import VisualServo

    servo = VisualServo(DEFAULT_CONFIG.servo, DEFAULT_CONFIG.turret, CAM)
    fire_px, splash_px = (480.0, 300.0), (500.0, 340.0)
    plain = servo.suppress_step(fire_px, splash_px, 10.0, 55.0)
    servo.reset()
    offset = servo.suppress_step(fire_px, splash_px, 10.0, 55.0, setpoint_offset_px=(0.0, 0.0))
    assert plain == offset


def test_a_nonzero_setpoint_offset_moves_the_target_row() -> None:
    """And that it is a real generalisation, not a no-op parameter."""
    from fireturret.control.servo import VisualServo

    servo = VisualServo(DEFAULT_CONFIG.servo, DEFAULT_CONFIG.turret, CAM)
    fire_px, splash_px = (480.0, 300.0), (480.0, 340.0)
    plain = servo.suppress_step(fire_px, splash_px, 0.0, 55.0)
    servo.reset()
    shifted = servo.suppress_step(fire_px, splash_px, 0.0, 55.0, setpoint_offset_px=(0.0, 40.0))
    assert shifted.range_error_px != plain.range_error_px


def test_the_fire_profile_passes_the_public_conformance_suite() -> None:
    check_profile(FireProfile())


# --------------------------------------------------------- the second profile

def test_washdown_covers_its_sector_and_then_stops() -> None:
    """A genuinely different shape of job: open-loop area coverage rather than
    closed-loop point targeting. If the core can express both, the abstraction
    is real."""
    sweep = SweepEngagement(sector_deg=(-30.0, 30.0), swath_deg=6.0)
    assert sweep.total_swaths() == 10
    assert sweep.is_complete(None, 0.0) is False

    while (aim := sweep.next_aim_deg()) is not None:
        sweep.note_wetted(aim)
    assert sweep.coverage_fraction() == pytest.approx(1.0)
    assert sweep.is_complete(None, 0.0) is True


def test_coverage_is_tracked_by_swept_azimuth_not_elapsed_time() -> None:
    """Time is a proxy that stops being true the moment the pan rate changes."""
    sweep = SweepEngagement()
    assert sweep.is_complete(None, elapsed_s=10_000.0) is False


def test_washdown_narrows_the_pump_limit() -> None:
    assert WashdownProfile().constraints.max_pump_pct == 60.0


def test_washdown_uses_a_fan_so_the_deposition_check_is_load_bearing() -> None:
    """With a wide fan the wetted footprint is much broader than the aim, which
    is what makes SP5's deposition keep-out matter rather than being academic."""
    profile = WashdownProfile(fan_angle_deg=25.0)
    assert not profile.spray_envelope.is_point
    assert profile.spray_envelope.wetted_half_width_m(7.0) > 0.5


def test_washdown_passes_the_conformance_suite() -> None:
    check_profile(WashdownProfile())


# ------------------------------------------------------------ the plugin API

def test_discovery_lists_the_builtins() -> None:
    names = {p.name for p in discover(GROUP_PROFILE)}
    assert {"fire", "washdown"} <= names
    assert {"null", "sim", "serial"} <= {p.name for p in discover(GROUP_RIG)}
    assert {"classical", "onnx"} <= {p.name for p in discover(GROUP_DETECTOR)}


def test_discovery_does_not_import_anything(monkeypatch) -> None:
    """Listing what is installed must not execute third-party code. Plugin
    authors really do write `sys.exit()` and network calls at module scope."""
    import importlib.metadata as md

    class Exploding:
        name = "boom"
        value = "nope:nope"
        dist = None

        def load(self):  # pragma: no cover - must never be called
            raise AssertionError("discovery imported a plugin")

    monkeypatch.setattr(md, "entry_points", lambda **kw: [Exploding()])
    found = discover(GROUP_PROFILE)
    assert any(p.name == "boom" for p in found)


def test_the_dogfood_rule_every_builtin_loads_through_the_api() -> None:
    """If the built-ins cannot use the extension point, it is not an extension
    point — it is a hole with documentation."""
    assert isinstance(load(GROUP_PROFILE, "fire"), MissionProfile)
    assert isinstance(load(GROUP_PROFILE, "washdown"), MissionProfile)
    assert load(GROUP_RIG, "null") is not None
    assert load(GROUP_RIG, "sim") is not None
    assert load(GROUP_RIG, "serial") is not None  # the class, not an open port
    assert load(GROUP_DETECTOR, "classical") is not None
    assert load(GROUP_DETECTOR, "onnx") is not None


def test_loading_a_rig_does_not_open_a_serial_port() -> None:
    """Discovery and loading must never actuate anything. The serial rig is
    returned as a CLASS for exactly this reason."""
    from fireturret.rig.serial_rig import SerialRig

    assert load(GROUP_RIG, "serial") is SerialRig


def test_a_plugin_cannot_shadow_a_builtin(monkeypatch) -> None:
    """`pip install evil-plugin` registering `fire` must not replace the
    flagship profile."""
    import importlib.metadata as md

    class Impostor:
        name = "fire"
        value = "evil:profile"
        dist = None

        def load(self):  # pragma: no cover
            raise AssertionError("a plugin shadowed a built-in")

    monkeypatch.setattr(md, "entry_points", lambda **kw: [Impostor()])
    assert [p for p in discover(GROUP_PROFILE) if p.name == "fire"][0].builtin is True
    assert load(GROUP_PROFILE, "fire").name == "fire"


@pytest.mark.parametrize("failure", [
    ImportError("no module named 'nope'"),
    SystemExit(1),                 # NOT an Exception
    KeyboardInterrupt(),           # NOT an Exception
    RuntimeError("exploded on the third frame"),
])
def test_a_broken_plugin_raises_PluginError_and_does_not_kill_the_process(
    monkeypatch, failure
) -> None:
    """`except BaseException`, deliberately: a module-scope `sys.exit()` raises
    SystemExit, which is not an Exception, and must not take the turret's control
    process down with it."""
    import importlib.metadata as md

    class Broken:
        name = "broken"
        value = "broken:thing"
        dist = None

        def load(self):
            raise failure

    monkeypatch.setattr(md, "entry_points", lambda **kw: [Broken()])
    with pytest.raises(PluginError, match="failed to load"):
        load(GROUP_PROFILE, "broken")


def test_a_plugin_that_fails_to_construct_is_also_contained(monkeypatch) -> None:
    import importlib.metadata as md

    class BadFactory:
        name = "bad"
        value = "bad:thing"
        dist = None

        def load(self):
            def factory():
                raise ValueError("cannot build")

            return factory

    monkeypatch.setattr(md, "entry_points", lambda **kw: [BadFactory()])
    with pytest.raises(PluginError, match="failed to construct"):
        load(GROUP_PROFILE, "bad")


def test_an_unknown_plugin_name_is_a_clear_error() -> None:
    with pytest.raises(PluginError, match="no plugin named"):
        load(GROUP_PROFILE, "does_not_exist")


@pytest.mark.parametrize("group", GROUPS)
def test_every_group_has_at_least_one_builtin(group) -> None:
    assert builtin_names(group), f"group {group} ships nothing, so nothing dogfoods it"


def test_an_unknown_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown plugin group"):
        discover("fireturret.nonsense.v0")


# --------------------------------------------------- the conformance suite

def test_the_shipped_splash_detector_passes_its_own_conformance_suite() -> None:
    """The suite is used BY the built-in tests, not just published. A conformance
    suite the maintainers do not run against their own implementation is
    documentation, and it drifts the first time an interface changes."""
    from fireturret.vision.impact import SplashDetector

    check_impact_observer(SplashDetector(min_area_px=10))


def test_the_overlap_case_catches_a_bbox_excluding_observer() -> None:
    """**Invariant 4, executable.** An observer that excluded the target's bbox
    would pass every obvious test and fail here — going blind exactly when the
    water is on target. This project has had that bug."""
    from fireturret.vision.impact import SplashDetector

    class BboxExcludingObserver(SplashDetector):
        def observe(self, frame_bgr, target_px, exclude_bbox=None):
            if exclude_bbox is not None:
                x, y, w, h = exclude_bbox
                frame_bgr = frame_bgr.copy()
                frame_bgr[y:y + h, x:x + w] = 0  # the tempting, wrong optimisation
            return super().observe(frame_bgr, target_px, exclude_bbox)

    with pytest.raises(ConformanceFailure, match="blind"):
        check_impact_observer(BboxExcludingObserver(min_area_px=10))


def test_the_shipped_rigs_pass_the_rig_conformance_suite() -> None:
    from fireturret.rig.interface import NullRig

    check_rig(NullRig())
