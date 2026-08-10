# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP7 — environment sensing and wind feedforward.

The load-bearing test in this file is the one that asserts the system CANNOT do
something: a constant wind, with no anemometer, must decay toward zero rather
than being absorbed as permanent wind. Constant wind and constant model bias are
not separately identifiable from splash data, and a filter that pretended
otherwise would confidently attribute a mounting error to the weather.
"""

from __future__ import annotations

import json

import pytest

from fireturret.adapt.environment import (
    RHO_REF,
    Environment,
    FileEnvironment,
    NullEnvironment,
    SerialEnvironment,
    saturation_vapour_pressure_pa,
)
from fireturret.adapt.wind import (
    PlantGainValidator,
    WindFilter,
    crosswind_drift_m,
)
from fireturret.ballistics import arc_with_time, exit_velocity, simulate_arc
from fireturret.config import DEFAULT_CONFIG

JET = DEFAULT_CONFIG.jet
TILT = DEFAULT_CONFIG.servo.suppress_tilt_deg


# ------------------------------------------------------- zero wind is identity

def test_zero_wind_is_bit_identical_to_the_old_groundspeed_drag() -> None:
    """The wind term is an extension, not a re-derivation. Every existing fixture
    depends on this being exact rather than close."""
    speed = exit_velocity(55.0, JET)
    assert simulate_arc(speed, TILT, JET) == arc_with_time(
        speed, TILT, JET, wind=(0.0, 0.0)
    ).points


def test_an_explicit_drag_override_of_none_changes_nothing() -> None:
    speed = exit_velocity(70.0, JET)
    assert (
        arc_with_time(speed, TILT, JET).points
        == arc_with_time(speed, TILT, JET, drag_k=None).points
    )


def test_a_headwind_shortens_the_throw_and_a_tailwind_lengthens_it() -> None:
    speed = exit_velocity(55.0, JET)
    still = arc_with_time(speed, TILT, JET).range_m
    head = arc_with_time(speed, TILT, JET, wind=(-4.0, 0.0)).range_m
    tail = arc_with_time(speed, TILT, JET, wind=(4.0, 0.0)).range_m
    assert head < still < tail


# ---------------------------------------------------------------- air density

def test_standard_air_is_the_reference_density() -> None:
    assert Environment().air_density == pytest.approx(RHO_REF, rel=0.01)
    assert Environment().density_ratio == pytest.approx(1.0, rel=0.01)


def test_hot_thin_air_reduces_drag_proportionally() -> None:
    """`k = rho C_d A / 2m` is LINEAR in density, so this is exact rather than
    fitted. Computed: 35 C at 84 kPa is 0.775 of standard — about 22% less drag,
    which at a 7 m throw is a real range change."""
    hot_high = Environment(temperature_c=35.0, pressure_pa=84_000.0)
    assert hot_high.density_ratio == pytest.approx(0.775, abs=0.01)
    assert hot_high.effective_drag_k(0.01) == pytest.approx(0.01 * hot_high.density_ratio)


def test_moist_air_is_LESS_dense_than_dry_air() -> None:
    """Surprises people often enough to be worth a test: water vapour is lighter
    than the nitrogen and oxygen it displaces."""
    dry = Environment(temperature_c=30.0, relative_humidity=0.0)
    humid = Environment(temperature_c=30.0, relative_humidity=1.0)
    assert humid.air_density < dry.air_density


def test_saturation_vapour_pressure_matches_known_values() -> None:
    assert saturation_vapour_pressure_pa(20.0) == pytest.approx(2339.0, rel=0.02)
    assert saturation_vapour_pressure_pa(0.0) == pytest.approx(611.2, rel=0.02)


@pytest.mark.parametrize("kwargs,exc", [
    ({"temperature_c": -300.0}, ValueError),
    ({"pressure_pa": 0.0}, ValueError),
    ({"relative_humidity": 1.5}, ValueError),
    ({"temperature_c": "hot"}, TypeError),
])
def test_nonsense_conditions_are_rejected_at_construction(kwargs, exc) -> None:
    """At construction, not at first use: these values come from a JSON file a
    weather station wrote, so the providers' error handling can only degrade to
    STANDARD if the failure happens where they can catch it."""
    with pytest.raises(exc):
        Environment(**kwargs)


# ------------------------------------------------------------------ providers

def test_the_null_provider_is_standard_air_and_says_it_has_no_anemometer() -> None:
    env = NullEnvironment().read()
    assert env.density_ratio == pytest.approx(1.0, rel=0.01)
    assert env.wind_speed_ms == 0.0
    assert env.anemometer is False


def test_the_file_provider_reads_what_a_weather_station_wrote(tmp_path) -> None:
    path = tmp_path / "weather.json"
    path.write_text(json.dumps({
        "temperature_c": 28.0, "pressure_pa": 99_000.0,
        "wind_east_ms": 3.0, "anemometer": True,
    }))
    env = FileEnvironment(path).read()
    assert env.temperature_c == 28.0
    assert env.wind_east_ms == 3.0
    assert env.anemometer is True


@pytest.mark.parametrize("content", ["", "not json", "[1,2,3]", '{"temperature_c": "hot"}'])
def test_a_broken_environment_file_degrades_to_standard(tmp_path, content) -> None:
    """A turret must not fail to run because a weather station rebooted."""
    path = tmp_path / "weather.json"
    path.write_text(content)
    assert FileEnvironment(path).read().density_ratio == pytest.approx(1.0, rel=0.01)


def test_a_missing_environment_file_degrades_to_standard(tmp_path) -> None:
    assert FileEnvironment(tmp_path / "nope.json").read() == Environment()


def test_unknown_keys_are_ignored_not_fatal(tmp_path) -> None:
    path = tmp_path / "weather.json"
    path.write_text(json.dumps({"temperature_c": 22.0, "from_a_later_version": 1}))
    assert FileEnvironment(path).read().temperature_c == 22.0


def test_the_serial_provider_degrades_when_the_pod_is_silent() -> None:
    class DeadPort:
        def readline(self):
            raise OSError("unplugged")

    provider = SerialEnvironment("COM9", serial_factory=lambda *a, **k: DeadPort())
    assert provider.read() == Environment()


def test_the_serial_provider_parses_a_line_of_json() -> None:
    class Pod:
        def readline(self):
            return b'{"temperature_c": 19.0, "wind_north_ms": 2.5, "anemometer": true}'

    provider = SerialEnvironment("COM9", serial_factory=lambda *a, **k: Pod())
    env = provider.read()
    assert env.temperature_c == 19.0
    assert env.wind_north_ms == 2.5


# ------------------------------------------------- THE identifiability statement

def test_a_constant_wind_with_no_anemometer_decays_toward_zero() -> None:
    """**The honest property, asserted rather than only documented.**

    Constant wind and constant model bias produce identical splash observations.
    A filter that absorbed a persistent offset as permanent wind would be
    confidently attributing a mounting error to the weather. The mean-reverting
    prior drains it instead, leaving it visible to SP11's learner — where a
    persistent, repeatable offset actually belongs.
    """
    f = WindFilter(has_anemometer=False)
    for _ in range(50):
        f.update_from_residual(5.0, 0.0)  # a steady 5 m/s "wind" from splash misses
        f.predict(dt=1.0)
    settled = f.state.east_ms

    for _ in range(200):
        f.predict(dt=1.0)  # observations stop; only the prior acts
    assert abs(f.state.east_ms) < abs(settled) * 0.05, (
        "a constant residual was absorbed as permanent wind"
    )


def test_an_anemometer_breaks_the_ambiguity_and_the_wind_persists() -> None:
    """A direct measurement observes the quantity itself rather than one of three
    things that could explain a miss, so it is not drained away."""
    f = WindFilter()
    for _ in range(20):
        f.update_from_anemometer(5.0, 0.0)
        f.predict(dt=1.0)
    assert f.state.east_ms == pytest.approx(5.0, abs=0.5)

    for _ in range(100):
        f.predict(dt=1.0)
    assert f.state.east_ms == pytest.approx(5.0, abs=0.5), "measured wind was decayed away"


def test_the_filter_converges_toward_a_residual_it_is_shown() -> None:
    f = WindFilter()
    for _ in range(30):
        f.update_from_residual(3.0, -2.0)
    assert f.state.east_ms == pytest.approx(3.0, abs=0.6)
    assert f.state.north_ms == pytest.approx(-2.0, abs=0.6)


def test_variance_shrinks_with_evidence_and_grows_without_it() -> None:
    f = WindFilter()
    start = f.state.variance
    f.update_from_residual(1.0, 0.0)
    assert f.state.variance < start
    shrunk = f.state.variance
    f.predict(dt=10.0)
    assert f.state.variance > shrunk


# ------------------------------------------------------------- crosswind drift

def test_crosswind_drift_uses_the_lag_time_rule() -> None:
    """`drift = w (t_f - r/v0)`. The bracket is how much longer the water took
    than a drag-free shot would have."""
    speed = exit_velocity(55.0, JET)
    arc = arc_with_time(speed, TILT, JET)
    drift = crosswind_drift_m(3.0, arc.flight_time_s, arc.range_m, speed)
    lag = arc.flight_time_s - arc.range_m / speed
    assert drift == pytest.approx(3.0 * lag)
    assert lag > 0.0, "a real jet is slower than a drag-free one"


def test_no_drag_means_no_drift() -> None:
    """The correct limit: with zero lag time there is no crosswind drift, however
    hard the wind blows."""
    assert crosswind_drift_m(10.0, flight_time_s=1.0, range_m=10.0, muzzle_speed_ms=10.0) == 0.0


def test_a_degenerate_muzzle_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        crosswind_drift_m(3.0, 1.0, 7.0, 0.0)


# --------------------------------------------------------- monitor-only, only

def test_the_plant_validator_only_advises() -> None:
    """The spec's replacement for online gain tuning. Online tuning would give an
    estimator write access to a safety loop, with a stability target derived from
    delay-free algebra on a loop that has a measured 1-2 sample delay. This
    raises an advisory and takes no action."""
    v = PlantGainValidator()
    for _ in range(10):
        v.observe(predicted_px=10.0, observed_px=30.0)
    assert v.disagreement_rate == 1.0
    assert "check calibration" in v.advisory()

    api = set(dir(PlantGainValidator))
    assert not {"apply", "set_gain", "correct", "tune"} & api, (
        "the validator grew a way to act; it is monitor-only by design"
    )


def test_agreement_raises_nothing() -> None:
    v = PlantGainValidator()
    for _ in range(10):
        v.observe(predicted_px=10.0, observed_px=10.5)
    assert v.advisory() is None


def test_a_near_zero_prediction_is_not_a_disagreement() -> None:
    """Dividing by a prediction of zero would make every tiny response look like
    an infinite error."""
    v = PlantGainValidator()
    assert v.observe(predicted_px=0.0, observed_px=5.0) is False
