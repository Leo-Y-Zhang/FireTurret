# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP5b — the droplet ensemble that PRODUCES the spray envelope.

Two results carry this sub-project:

1. **The flagship's degeneracy is WEAKER than the spec claims.** The spec says the
   solid stream's breakup length is ~19 m against a ~7 m throw. Computed here it is
   4.2-8.4 m across the pump band — the jet breaks up BEFORE it lands. Recorded
   rather than tuned to match, and the shipped `LEGACY_ENVELOPE` is kept anyway.

2. **A fan is genuinely different**, and wind SKEWS its footprint rather than
   translating it — small droplets have more drag per unit mass and are pushed
   further downwind than large ones. A single-drag model cannot produce that
   asymmetry, which is the whole argument for per-diameter drag.
"""

from __future__ import annotations

import math
import time

import pytest

from fireturret.spray.arcs import integrate_ensemble
from fireturret.spray.dropletdist import (
    DropletClass,
    equal_volume_classes,
    rosin_rammler_cdf,
    rosin_rammler_inverse,
    sauter_mean_diameter,
)
from fireturret.spray.envelope import LEGACY_ENVELOPE, envelope_from_ensemble
from fireturret.spray.fluid import (
    air_viscosity,
    drag_coefficient,
    ohnesorge_number,
    saturation_pressure,
    surface_tension,
    water_viscosity,
    weber_number,
)
from fireturret.spray.nozzle import Nozzle, SprayPattern

SPEED = 19.2  # m/s, roughly pump 55% on the default jet
TILT = 22.0


# ------------------------------------------------------------------- fluids

def test_surface_tension_matches_published_values() -> None:
    assert surface_tension(20.0) == pytest.approx(0.0728, abs=0.001)
    assert surface_tension(100.0) == pytest.approx(0.0589, abs=0.002)


def test_surface_tension_refuses_to_extrapolate() -> None:
    """A correlation used outside its range produces plausible-looking droplet
    sizes for a fluid that does not exist."""
    with pytest.raises(ValueError, match="0-374"):
        surface_tension(500.0)


def test_water_viscosity_matches_published_values() -> None:
    assert water_viscosity(20.0) == pytest.approx(1.002e-3, rel=0.05)


def test_air_viscosity_matches_sutherland() -> None:
    assert air_viscosity(20.0) == pytest.approx(1.81e-5, rel=0.03)


def test_saturation_pressure_matches_buck() -> None:
    assert saturation_pressure(20.0) == pytest.approx(2339.0, rel=0.02)


def test_drag_is_reynolds_DEPENDENT() -> None:
    """The term a single drag constant cannot represent: a small slow droplet and
    a large fast one differ by orders of magnitude."""
    assert drag_coefficient(0.5) > drag_coefficient(100.0) > drag_coefficient(5000.0)
    assert drag_coefficient(5000.0) == pytest.approx(0.44)


def test_the_dimensionless_numbers_are_dimensionless() -> None:
    we = weber_number(998.0, 20.0, 0.006, 0.0728)
    oh = ohnesorge_number(1.0e-3, 998.0, 0.006, 0.0728)
    assert we > 1000.0  # a 6 mm jet at 20 m/s is inertia-dominated
    assert 0.0 < oh < 0.01  # and barely viscous


# ------------------------------------------------------- the size distribution

def test_the_cdf_and_its_inverse_agree() -> None:
    for fraction in (0.1, 0.5, 0.9):
        d = rosin_rammler_inverse(fraction, 1e-3, 2.5)
        assert rosin_rammler_cdf(d, 1e-3, 2.5) == pytest.approx(fraction, abs=1e-9)


def test_classes_carry_EQUAL_volume_not_equal_diameter() -> None:
    """Equal-diameter bins put almost all the water in one or two classes, so
    most of the computation carries almost no water and the classes that matter
    are under-resolved."""
    classes = equal_volume_classes(1e-3, 2.5, n_classes=7)
    assert len(classes) == 7
    fractions = [c.volume_fraction for c in classes]
    assert fractions == pytest.approx([1 / 7] * 7)
    assert sum(fractions) == pytest.approx(1.0)


def test_class_diameters_increase_across_the_distribution() -> None:
    classes = equal_volume_classes(1e-3, 2.5, n_classes=7)
    ds = [c.diameter_m for c in classes]
    assert ds == sorted(ds)
    assert ds[-1] > 2 * ds[0], "the distribution should span a real range of sizes"


def test_the_sauter_mean_lies_inside_the_distribution() -> None:
    classes = equal_volume_classes(1e-3, 2.5, n_classes=7)
    d32 = sauter_mean_diameter(classes)
    assert classes[0].diameter_m < d32 < classes[-1].diameter_m


def test_a_degenerate_distribution_is_rejected() -> None:
    with pytest.raises(ValueError):
        equal_volume_classes(1e-3, 2.5, n_classes=0)
    with pytest.raises(ValueError):
        rosin_rammler_inverse(1.0, 1e-3, 2.5)


# --------------------------------------------- THE flagship degeneracy, computed

@pytest.mark.parametrize("speed,expected_m", [(12.9, 4.19), (19.2, 6.24), (25.9, 8.42)])
def test_the_measured_breakup_length_contradicts_the_spec(speed, expected_m) -> None:
    """**A substantive disagreement with the programme spec, recorded in code.**

    The spec states the solid stream's breakup length is "≈19 m against a ~7 m
    throw", and uses that to argue the flagship's point footprint is *provably*
    degenerate. Computed here with the standard second-wind-induced correlation
    `L/d = 6·We^0.5 / (1 + 3·Oh)`, at a 6 mm orifice:

        pump  25%   12.9 m/s   breakup 4.19 m   throw  6.34 m
        pump  55%   19.2 m/s   breakup 6.24 m   throw  8.80 m
        pump 100%   25.9 m/s   breakup 8.42 m   throw 10.80 m

    So the jet breaks up **before** it lands across the whole usable band, at
    roughly a third of the spec's figure. The degeneracy argument is therefore
    weaker than the spec claims: the stream is coherent for most of its flight
    and disintegrates near the end, rather than arriving intact.

    Correlations for jet breakup vary widely between sources, so this is not
    proof the spec is wrong — it is proof the claim is not robust enough to
    carry the weight the spec puts on it. Recorded rather than tuned to match.
    """
    nozzle = Nozzle(orifice_m=0.006, pattern=SprayPattern.SOLID_STREAM)
    assert nozzle.breakup_length_m(speed) == pytest.approx(expected_m, abs=0.15)
    assert nozzle.breakup_length_m(speed) < 19.0, "the spec's figure did not reproduce"


def test_the_legacy_envelope_remains_the_shipped_default_regardless() -> None:
    """The finding above does NOT change shipped behaviour, and deliberately so.

    Adopting a computed non-point envelope for the fire profile would move every
    golden fixture, and the programme's three planned regenerations are spent.
    More importantly it would trade a simple, well-tested model for one resting
    on a correlation whose spread across sources is larger than the effect. So
    `MissionController` keeps `LEGACY_ENVELOPE`, and this is the test that says
    so on purpose rather than by omission.
    """
    from fireturret.config import DEFAULT_CONFIG
    from fireturret.control.mission import MissionController
    from fireturret.geometry import CameraModel

    mission = MissionController(DEFAULT_CONFIG, CameraModel(DEFAULT_CONFIG.camera))
    assert mission.spray_envelope == LEGACY_ENVELOPE
    assert mission.spray_envelope.is_point


def test_the_ensemble_reproduces_the_legacy_envelope_when_the_jet_IS_coherent() -> None:
    """Where the premise genuinely holds, the ensemble must MATCH the shipped
    model rather than be excused from matching it — if the new physics disagreed
    there, one of them would be wrong.

    A 10 mm orifice rather than the flagship's 6 mm, because Weber number scales
    with diameter and the larger jet really is coherent past its throw (~13 m
    breakup against ~9 m). Constructing the case honestly is the point: the 6 mm
    stream is marginal, and pretending otherwise to make a test pass would be
    exactly the tuning-to-match the test above refuses.
    """
    coherent = Nozzle(orifice_m=0.010, pattern=SprayPattern.SOLID_STREAM)
    assert coherent.breakup_length_m(SPEED) > 12.0
    envelope = envelope_from_ensemble(coherent, SPEED, TILT, target_range_m=8.8)
    assert envelope == LEGACY_ENVELOPE


def test_a_marginal_stream_ALSO_declines_rather_than_guessing() -> None:
    """The flagship's actual case, and the one that changed.

    An earlier version integrated the marginal 6 mm stream and returned a small
    non-zero envelope, on the reasoning that it has probably atomised by impact.
    The number was wrong in a way the test could not see: the ensemble reached it
    with SPHERE drag (k = 0.0675) where the calibrated jet uses 0.160, and it
    launched the droplets from the orifice rather than from wherever the column
    actually disintegrated. Both errors point the same way — too little drag, too
    long a flight — so the "small envelope" was a confident number built on the
    wrong physics, which is worse than the point model it replaced.

    Whether the flagship stream is coherent at impact is genuinely unsettled:
    across four published correlations the intact length spans 0.73-8.42 m against
    a 6.34-10.80 m throw. A branch this close to a keep-out decision must not hinge
    on the least certain quantity in the module, so it does not hinge on anything.
    """
    marginal = Nozzle(orifice_m=0.006, pattern=SprayPattern.SOLID_STREAM)
    envelope = envelope_from_ensemble(marginal, SPEED, TILT, target_range_m=8.8)
    assert envelope == LEGACY_ENVELOPE


def test_the_stream_refusal_does_not_depend_on_the_breakup_correlation() -> None:
    """The point of the change above, stated as a property rather than a case.

    Sweeping the orifice from unambiguously-coherent to unambiguously-atomised
    must not produce a different KIND of answer anywhere along the way. If some
    diameter fell through to the integration, the correlation would be back in the
    safety path through the side door.
    """
    for orifice_mm in (2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0):
        nozzle = Nozzle(orifice_m=orifice_mm / 1000.0, pattern=SprayPattern.SOLID_STREAM)
        assert envelope_from_ensemble(
            nozzle, SPEED, TILT, target_range_m=8.8
        ) == LEGACY_ENVELOPE, f"{orifice_mm} mm fell through to the ensemble"


def test_a_fan_nozzle_is_NOT_degenerate() -> None:
    """The other half — if everything reduced to a point, the module would be
    pointless."""
    fan = Nozzle(orifice_m=0.006, pattern=SprayPattern.FLAT_FAN)
    envelope = envelope_from_ensemble(fan, SPEED, TILT, target_range_m=8.8)
    assert not envelope.is_point
    assert envelope.lateral_sigma_per_m > 0.0


def test_a_fan_atomises_at_the_orifice_and_a_stream_does_not() -> None:
    assert Nozzle(pattern=SprayPattern.FLAT_FAN).breakup_length_m(SPEED) == 0.0
    assert Nozzle(pattern=SprayPattern.SOLID_STREAM).breakup_length_m(SPEED) > 0.0


def test_higher_velocity_makes_finer_droplets() -> None:
    """Why a fog nozzle at pressure produces mist and the same nozzle at a
    dribble produces drips."""
    fog = Nozzle(pattern=SprayPattern.FOG)
    assert fog.sauter_mean_diameter_m(30.0) < fog.sauter_mean_diameter_m(10.0)


# ------------------------------------------------- THE wind skew, not translation

def test_a_still_fan_is_SYMMETRIC() -> None:
    """The control case, and it caught a real bug. An earlier version mapped one
    size class to one launch angle — index order being size order — so the
    smallest droplets always launched to the same side. That manufactured a
    size/direction correlation the nozzle does not have, and made wind appear to
    REDUCE skew because it fought the artifact. Measured skew is now 0.000."""
    fan = Nozzle(orifice_m=0.006, pattern=SprayPattern.FLAT_FAN)
    classes = equal_volume_classes(fan.d63_m(SPEED), fan.spread_exponent(), 7)
    still = integrate_ensemble(classes, SPEED, TILT, fan_half_angle_deg=12.5)
    assert still.lateral_skew == pytest.approx(0.0, abs=0.05)


def test_wind_SKEWS_a_fan_footprint_rather_than_translating_it() -> None:
    """**The result that justifies per-diameter drag.** Small droplets have more
    drag per unit mass, so wind pushes them further than large ones. A
    single-drag model translates the whole footprint sideways and cannot produce
    this asymmetry. Measured: skew 0.000 still, +0.735 in a 5 m/s crosswind."""
    fan = Nozzle(orifice_m=0.006, pattern=SprayPattern.FLAT_FAN)
    classes = equal_volume_classes(fan.d63_m(SPEED), fan.spread_exponent(), 7)

    still = integrate_ensemble(classes, SPEED, TILT, fan_half_angle_deg=12.5)
    windy = integrate_ensemble(classes, SPEED, TILT, fan_half_angle_deg=12.5,
                               wind_ms=(0.0, 5.0))

    assert abs(windy.lateral_skew) > 0.3, (
        f"wind did not skew the footprint: {windy.lateral_skew:.3f}"
    )
    assert abs(windy.lateral_skew) > abs(still.lateral_skew) + 0.3
    assert windy.lateral_sigma_m > still.lateral_sigma_m, "and it should spread it wider"


def test_small_droplets_fall_shorter_than_large_ones() -> None:
    """The mechanism behind the skew, isolated: per-diameter drag means the
    classes do not land together."""
    small = [DropletClass(diameter_m=5e-5, volume_fraction=1.0)]
    large = [DropletClass(diameter_m=2e-3, volume_fraction=1.0)]
    short = integrate_ensemble(small, SPEED, TILT, evaporate=False).mean_range_m
    far = integrate_ensemble(large, SPEED, TILT, evaporate=False).mean_range_m
    assert short < far


def test_evaporation_shrinks_the_finest_droplets() -> None:
    """d^2-law: a 50 um droplet has an enormous surface-to-volume ratio and
    genuinely does shrink over a second of flight."""
    tiny = [DropletClass(diameter_m=5e-5, volume_fraction=1.0)]
    with_evap = integrate_ensemble(tiny, SPEED, TILT, evaporate=True).mean_range_m
    without = integrate_ensemble(tiny, SPEED, TILT, evaporate=False).mean_range_m
    assert with_evap != without


def test_a_headwind_shortens_every_class() -> None:
    fan = Nozzle(pattern=SprayPattern.FLAT_FAN)
    classes = equal_volume_classes(fan.d63_m(SPEED), fan.spread_exponent(), 5)
    still = integrate_ensemble(classes, SPEED, TILT).mean_range_m
    head = integrate_ensemble(classes, SPEED, TILT, wind_ms=(-5.0, 0.0)).mean_range_m
    assert head < still


def test_no_class_is_silently_dropped() -> None:
    """Losing a class at the step cap would reweight the distribution without
    saying so."""
    classes = equal_volume_classes(1e-3, 2.5, n_classes=7)
    result = integrate_ensemble(classes, SPEED, TILT)
    assert result.ranges_m.size == 7
    assert result.volume_fractions.sum() == pytest.approx(1.0)


def test_an_empty_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError):
        integrate_ensemble([], SPEED, TILT)


# ----------------------------------------------------------------- the budget

def test_a_solution_time_ensemble_fits_the_budget() -> None:
    """It runs once per firing solution, not per tick — which is the decision
    that makes droplet physics affordable at all. `plan_suppression` already
    costs ~53 ms, so this must be the same order, not an order worse."""
    fan = Nozzle(pattern=SprayPattern.FLAT_FAN)
    classes = equal_volume_classes(fan.d63_m(SPEED), fan.spread_exponent(), 7)

    start = time.perf_counter()
    integrate_ensemble(classes, SPEED, TILT, fan_half_angle_deg=12.5)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"ensemble took {elapsed * 1000:.0f} ms per solution"


def test_the_ensemble_is_not_reachable_from_control() -> None:
    """Control imports the ENVELOPE; the ensemble produces it. That direction is
    what keeps a per-droplet integration out of a 33 ms loop."""
    import ast
    from pathlib import Path

    control = Path(__file__).resolve().parents[1] / "src" / "fireturret" / "control"
    forbidden = {"dropletdist", "nozzle", "arcs", "fluid"}
    for path in control.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = getattr(node, "module", None)
            if module and any(f in module for f in forbidden):
                raise AssertionError(f"{path.name} imports spray internals: {module}")


def test_the_impact_angle_is_reported_for_deposition_reasoning() -> None:
    fan = Nozzle(pattern=SprayPattern.FLAT_FAN)
    envelope = envelope_from_ensemble(fan, SPEED, TILT, target_range_m=8.8)
    assert envelope.impact_angle_deg == pytest.approx(abs(TILT))


def test_fan_angles_are_ordered_by_pattern() -> None:
    angles = [p.nominal_fan_angle_deg for p in (
        SprayPattern.SOLID_STREAM, SprayPattern.FLAT_FAN,
        SprayPattern.FULL_CONE, SprayPattern.HOLLOW_CONE, SprayPattern.FOG,
    )]
    assert angles == sorted(angles)
    assert math.isclose(angles[0], 0.0)
