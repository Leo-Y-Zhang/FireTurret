# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
from fireturret.rig.interface import RigCommand
from fireturret.rig.protocol import encode_command, parse_telemetry


def test_command_encoding_format() -> None:
    line = encode_command(RigCommand(-12.5, 30.0, 65.0, True, laser=False, warn=True))
    assert line == "C p=-12.50 t=30.00 w=65.0 v=1 l=0 x=1\n"


def test_telemetry_roundtrip() -> None:
    t = parse_telemetry("S p=-12.50 t=30.00 w=65.0 v=1 e=0\n")
    assert t is not None
    assert t.pan_deg == -12.5
    assert t.tilt_deg == 30.0
    assert t.pump_pct == 65.0
    assert t.valve is True
    assert t.estop is False
    assert t.ok is True


def test_estop_flag_parses() -> None:
    t = parse_telemetry("S p=0 t=20 w=0 v=0 e=1")
    assert t is not None and t.estop is True


def test_homed_flag_parses() -> None:
    unhomed = parse_telemetry("S p=0 t=20 w=0 v=0 e=0 h=0")
    assert unhomed is not None and unhomed.homed is False
    homed = parse_telemetry("S p=0 t=20 w=0 v=0 e=0 h=1")
    assert homed is not None and homed.homed is True
    # absent h ⇒ NOT homed. This field gates water on a known pan reference, so
    # an unknown value must fail CLOSED. Older firmware that never sends h= is
    # therefore treated as un-homed rather than silently trusted (SP1).
    legacy = parse_telemetry("S p=0 t=20 w=0 v=0 e=0")
    assert legacy is not None and legacy.homed is False


def test_malformed_lines_return_none() -> None:
    assert parse_telemetry("") is None
    assert parse_telemetry("garbage") is None
    assert parse_telemetry("S p=abc t=1 w=2 v=0") is None
    assert parse_telemetry("C p=1 t=2 w=3 v=0 l=0") is None
    assert parse_telemetry("S p=1") is None
