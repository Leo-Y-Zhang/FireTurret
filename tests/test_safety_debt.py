# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP1 — the other four verified places where the shipped system was weaker
than its own documentation claimed.

Each test here corresponds to a defect reproduced against the pre-SP1 HEAD:

1. `RigTelemetry.homed` defaulted True and an absent `h=` was read as homed —
   fail-OPEN on the one field that gates water on a known pan reference.
2. The wire protocol's firmware parser dispatches on the token's FIRST
   CHARACTER, so any two tokens sharing one would silently alias.
3. The firmware's 64-byte line buffer truncates silently; a command must fit.
4. `config_from_dict` raised on a missing group / unknown field, silently
   dropped unknown groups, and no schema version was ever written.
5. `make_server` defaulted to 0.0.0.0 — an unauthenticated MJPEG stream and a
   POST /estop that latches SAFE, reachable from anywhere on the network.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from fireturret.config import DEFAULT_CONFIG, config_from_dict, config_to_dict
from fireturret.rig.interface import RigCommand, RigTelemetry
from fireturret.rig.protocol import encode_command, parse_telemetry

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware" / "turret_firmware" / "turret_firmware.ino"


# ------------------------------------------------------------------ fail-closed

def test_homed_defaults_to_false() -> None:
    """A rig that does not report homing must not be assumed homed."""
    tel = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True)
    assert tel.homed is False


def test_absent_h_field_is_not_homed() -> None:
    """Old firmware that never sends h= must fail CLOSED, not open."""
    tel = parse_telemetry("S p=0.00 t=20.00 w=0.0 v=0 e=0")
    assert tel is not None
    assert tel.homed is False


def test_explicit_h_is_still_honoured() -> None:
    assert parse_telemetry("S p=0 t=20 w=0 v=0 e=0 h=1").homed is True
    assert parse_telemetry("S p=0 t=20 w=0 v=0 e=0 h=0").homed is False


def test_null_rig_declares_itself_homed() -> None:
    """NullRig echoes commands for dry runs over recorded video; it has no pan
    axis to home, so it must say so explicitly rather than inherit a default."""
    from fireturret.rig.interface import NullRig

    assert NullRig().telemetry().homed is True


# -------------------------------------------------------------- wire protocol

def _command_tokens() -> list[str]:
    line = encode_command(RigCommand(0.0, 0.0, 0.0, False))
    return [t.partition("=")[0] for t in line.strip().split()[1:]]


def test_no_two_command_tokens_share_a_first_character() -> None:
    """The firmware parser dispatches on tok[0]. Until that is fixed on every
    deployed board, two tokens sharing a first character silently alias — so
    this constraint is part of the protocol contract, not an implementation
    detail."""
    firsts = [t[0] for t in _command_tokens()]
    assert len(set(firsts)) == len(firsts), f"aliasing tokens: {_command_tokens()}"


def test_worst_case_command_fits_the_firmware_buffer() -> None:
    """A line longer than the firmware buffer is silently truncated, and a
    truncated 'close the valve' leaves water flowing. The host must never be
    able to emit one."""
    m = re.search(r"char\s+buf\[(\d+)\]", FIRMWARE.read_text(encoding="utf-8", errors="replace"))
    assert m, "could not find the firmware line buffer declaration"
    buf_size = int(m.group(1))

    worst = encode_command(
        RigCommand(pan_deg=-179.99, tilt_deg=-179.99, pump_pct=-100.0,
                   valve=True, laser=True, warn=True)
    )
    assert len(worst) < buf_size, f"{len(worst)} bytes vs buf[{buf_size}]"


def _firmware_code() -> str:
    """Firmware source with comments stripped.

    These are source-text assertions about what the firmware *does*. Comments
    routinely quote the defective code they replaced — that is good commenting
    and must not read as a regression, so the prose is removed before matching.
    """
    src = FIRMWARE.read_text(encoding="utf-8", errors="replace")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def test_firmware_matches_full_keys_not_first_characters() -> None:
    """The parser must compare the whole key up to '=', so a future token like
    'pr=' cannot hijack 'p='."""
    assert "tok[0] ==" not in _firmware_code(), (
        "firmware still dispatches on the first character"
    )


def test_firmware_does_not_seed_parse_locals_from_previous_state() -> None:
    """Parse locals seeded from the previous command mean a truncated line
    inherits the old valve state — water keeps flowing."""
    code = _firmware_code()
    for seed in ("= panTarget", "= tiltTarget", "= pumpPct", "= valveOpen",
                 "= laserOn", "= warnOn"):
        assert f"float p {seed}" not in code and f"int v {seed}" not in code, (
            f"firmware still seeds a parse local from prior state ({seed})"
        )
    # and positively: the locals must start from constants
    assert "float p = 0.0f" in code, "parse locals must start from safe constants"


def test_firmware_requires_every_control_field_before_acting() -> None:
    """A truncated line loses trailing tokens. Acting on the surviving prefix is
    what let a truncated 'close the valve' leave water flowing, so a line missing
    any control field must be refused outright."""
    code = _firmware_code()
    for flag in ("sawPan", "sawTilt", "sawPump", "sawValve"):
        assert flag in code, f"firmware does not track whether {flag[3:].lower()} arrived"
    assert re.search(r"if\s*\(!sawPan\s*\|\|\s*!sawTilt\s*\|\|\s*!sawPump\s*\|\|\s*!sawValve\)",
                     code), "firmware does not refuse a line missing a control field"


def test_firmware_only_feeds_the_watchdog_on_a_valid_line() -> None:
    """lastCommandMs must not be refreshed by a malformed/truncated line, or the
    heartbeat watchdog is fed by exactly the failure it exists to catch."""
    src = FIRMWARE.read_text(encoding="utf-8", errors="replace")
    assert "MALFORMED_LINE_DOES_NOT_FEED_WATCHDOG" in src, (
        "firmware must carry the marker comment documenting this guarantee"
    )


# ----------------------------------------------------------------- config I/O

def test_schema_version_round_trips() -> None:
    d = config_to_dict(DEFAULT_CONFIG)
    assert "schema_version" in d
    assert config_from_dict(d) == DEFAULT_CONFIG


def test_missing_group_falls_back_to_defaults() -> None:
    d = config_to_dict(DEFAULT_CONFIG)
    del d["turret"]
    assert config_from_dict(d).turret == DEFAULT_CONFIG.turret


def test_unknown_field_is_ignored_not_fatal() -> None:
    d = config_to_dict(DEFAULT_CONFIG)
    d["jet"]["bogus_field_from_the_future"] = 1
    assert config_from_dict(d).jet == DEFAULT_CONFIG.jet


def test_unknown_group_survives_a_round_trip() -> None:
    """Plugin config lives in groups this version does not know about. Dropping
    them silently destroys a third party's settings on every save."""
    d = config_to_dict(DEFAULT_CONFIG)
    d["some_plugin"] = {"threshold": 0.25}
    assert config_to_dict(config_from_dict(d))["some_plugin"] == {"threshold": 0.25}


def test_a_future_schema_version_warns() -> None:
    d = config_to_dict(DEFAULT_CONFIG)
    d["schema_version"] = 999
    with pytest.warns(UserWarning, match="newer"):
        config_from_dict(d)


# ------------------------------------------------------------------ web bind

def test_web_server_defaults_to_loopback() -> None:
    from fireturret.ui.webserver import make_server

    assert inspect.signature(make_server).parameters["host"].default == "127.0.0.1"


def test_non_loopback_bind_requires_acknowledgement() -> None:
    """Binding the unauthenticated console to a routable address exposes a
    remote E-stop. It must be a deliberate, explicit act."""
    from fireturret.ui.webserver import WebState, make_server

    with pytest.raises(ValueError, match="acknowledge"):
        make_server(WebState(), host="0.0.0.0", port=0)


def test_non_loopback_bind_is_allowed_when_acknowledged() -> None:
    from fireturret.ui.webserver import WebState, make_server

    srv = make_server(WebState(), host="127.0.0.1", port=0, acknowledge_exposure=True)
    srv.server_close()


def test_serve_also_defaults_to_loopback() -> None:
    """`make_server` is the chokepoint, but `serve` is what the CLI calls — a
    routable default there would reach the network before the guard ran."""
    from fireturret.ui.webserver import serve

    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


# ------------------------------------------------------------- CLI arming gate

def _parse(argv: list[str]):
    """Parse argv against the real CLI surface without dispatching a command."""
    from fireturret.__main__ import build_parser

    return build_parser().parse_args(argv)


def test_selftest_water_requires_the_arming_flag() -> None:
    """A bare `--water` flag used to pulse the pump with no typed confirmation,
    while `run` and `web` demanded --arm-water AND a typed ARM. A self-test's
    pump pulse is exactly as wet as a mission's."""
    with pytest.raises(SystemExit):
        _parse(["selftest", "--port", "COM3", "--water"])

    args = _parse(["selftest", "--port", "COM3", "--arm-water"])
    assert args.arm_water is True


def test_selftest_defaults_to_dry() -> None:
    args = _parse(["selftest", "--port", "COM3"])
    assert args.arm_water is False


def test_web_defaults_to_loopback_and_has_an_expose_flag() -> None:
    args = _parse(["web"])
    assert args.host == "127.0.0.1"
    assert args.expose is False
    assert _parse(["web", "--host", "0.0.0.0", "--expose"]).expose is True


def test_every_hardware_path_shares_one_rig_factory_and_one_arm_gate() -> None:
    """Three commands can actuate a real turret. If any of them builds its own
    SerialRig or inlines its own arming prompt, the gates drift apart — which is
    how `selftest` ended up with a bare --water flag in the first place."""
    import fireturret.__main__ as cli

    for name in ("_cmd_run", "_cmd_web", "_cmd_selftest"):
        src = inspect.getsource(getattr(cli, name))
        assert "SerialRig(" not in src, f"{name} constructs its own rig"
        assert "_confirm_arm()" not in src, f"{name} inlines its own arming prompt"

    assert "SerialRig(" in inspect.getsource(cli._open_serial_rig)
    assert "_confirm_arm()" in inspect.getsource(cli._arm_water)


def test_arm_gate_denies_without_the_flag(monkeypatch, capsys) -> None:
    import argparse

    import fireturret.__main__ as cli

    monkeypatch.setattr("builtins.input", lambda *_: "ARM")
    assert cli._arm_water(argparse.Namespace(arm_water=False)) is False
    assert "dry-aim" in capsys.readouterr().out


def test_arm_gate_denies_without_the_typed_word(monkeypatch) -> None:
    import argparse

    import fireturret.__main__ as cli

    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    assert cli._arm_water(argparse.Namespace(arm_water=True)) is False


def test_arm_gate_allows_flag_plus_typed_word(monkeypatch) -> None:
    import argparse

    import fireturret.__main__ as cli

    monkeypatch.setattr("builtins.input", lambda *_: "ARM")
    assert cli._arm_water(argparse.Namespace(arm_water=True)) is True


# ------------------------------------------------------- self-test geofencing

def test_selftest_never_aims_into_the_keepout() -> None:
    """The self-test drives the rig directly, bypassing MissionController and
    therefore its geofence. Bring-up is precisely when someone is standing next
    to the machine, so nothing here may point into the protected sector."""
    from dataclasses import replace as dc_replace

    from fireturret.control.mission import pan_touches_keepout
    from fireturret.rig.interface import NullRig
    from fireturret.selftest import run_selftest

    cfg = dc_replace(DEFAULT_CONFIG, turret=dc_replace(
        DEFAULT_CONFIG.turret, pan_keepout_deg=(-8.0, 8.0)))

    commanded: list[float] = []

    class RecordingRig(NullRig):
        def command(self, cmd):
            commanded.append(cmd.pan_deg)
            return super().command(cmd)

    run_selftest(RecordingRig(), cfg, water=True, dwell=0.0, sleep=lambda _s: None)

    assert commanded, "the self-test actuated nothing at all"
    offenders = [p for p in commanded if pan_touches_keepout(p, (-8.0, 8.0))]
    assert not offenders, f"self-test aimed into the keep-out at {offenders}"


def test_selftest_aborts_when_the_keepout_leaves_nowhere_safe() -> None:
    """A sector spanning the whole travel is a real configuration. Refusing to
    actuate is the only honest response — picking the least-bad angle is not."""
    from dataclasses import replace as dc_replace

    from fireturret.rig.interface import NullRig
    from fireturret.selftest import run_selftest

    cfg = dc_replace(DEFAULT_CONFIG, turret=dc_replace(
        DEFAULT_CONFIG.turret, pan_keepout_deg=(-170.0, 170.0)))

    report = run_selftest(NullRig(), cfg, water=True, dwell=0.0, sleep=lambda _s: None)
    assert report.passed is False
    assert any(name == "ABORTED" for name, _ok, _d in report.checks)


def test_selftest_without_a_keepout_is_unchanged() -> None:
    """The geofencing must be inert when no sector is configured — the default."""
    from fireturret.rig.interface import NullRig
    from fireturret.selftest import run_selftest

    report = run_selftest(NullRig(), DEFAULT_CONFIG, water=True, dwell=0.0,
                          sleep=lambda _s: None)
    assert report.passed is True
