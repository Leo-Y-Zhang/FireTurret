# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Cover SerialRig (reader thread, parsing, connect verification, staleness, safe
close, port errors) with a fake serial port — no hardware required."""

import time

import pytest

import triton.rig.serial_rig as sr_mod
from triton.rig.interface import RigCommand


class FakeSerial:
    """Minimal stand-in for serial.Serial that streams one telemetry line on
    repeat and records what the host writes."""

    def __init__(self, port, baudrate=115200, timeout=0.2):
        self.port = port
        self.written: list[bytes] = []
        self.closed = False
        self._line = b"S p=1.50 t=25.00 w=40.0 v=1 e=0\n"
        self._pos = 0

    def read(self, n):
        if self._pos >= len(self._line):
            self._pos = 0  # loop forever so telemetry stays fresh
        chunk = self._line[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def write(self, data):
        self.written.append(data)

    def close(self):
        self.closed = True


class SilentSerial(FakeSerial):
    def read(self, n):
        return b""  # never sends telemetry


class BrieflyTalks(FakeSerial):
    """Streams a couple of telemetry lines, then goes silent (to test staleness)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._budget = len(self._line) * 2

    def read(self, n):
        if self._budget <= 0:
            return b""
        chunk = super().read(n)
        self._budget -= len(chunk)
        return chunk


class WriteFailsSerial(FakeSerial):
    """Streams enough telemetry to connect, then goes silent; the command write
    raises as if the USB link dropped mid-mission."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._budget = len(self._line) * 3  # connect, then quiet (reader idle)

    def read(self, n):
        if self._budget <= 0:
            return b""
        chunk = super().read(n)
        self._budget -= len(chunk)
        return chunk

    def write(self, data):
        raise sr_mod.serial.SerialException("device disconnected")


def test_serial_rig_command_survives_write_failure_and_degrades_to_not_ok(monkeypatch):
    """A serial disconnect on the per-frame command write must NOT propagate an
    uncaught exception (which would abort the mission loop); instead the link is
    torn down so telemetry() reports ok=False and the mission's SAFE failsafe
    takes over — mirroring the read loop's graceful handling."""
    monkeypatch.setattr(sr_mod.serial, "Serial", WriteFailsSerial)
    rig = sr_mod.SerialRig("COM_FAKE", reset_delay=0.0)
    try:
        assert rig.telemetry().ok  # connected fine before the disconnect
        time.sleep(0.05)  # reader drains its budget and goes idle
        rig.command(RigCommand(10.0, 30.0, 55.0, True))  # must not raise
        assert rig.telemetry().ok is False  # degraded failsafe, not a crash
    finally:
        rig.close()


def test_serial_rig_parses_telemetry_and_writes_commands(monkeypatch):
    monkeypatch.setattr(sr_mod.serial, "Serial", FakeSerial)
    rig = sr_mod.SerialRig("COM_FAKE", reset_delay=0.0)
    try:
        tel = rig.telemetry()
        assert tel.ok  # fresh data arrived via the reader thread
        assert tel.pan_deg == 1.5
        assert tel.tilt_deg == 25.0
        assert tel.valve is True
        rig.command(RigCommand(10.0, 30.0, 55.0, True))
        assert rig._serial.written[-1].startswith(b"C ")
    finally:
        rig.close()


def test_serial_rig_raises_without_telemetry(monkeypatch):
    # a port that opens but never talks (wrong device / baud / no firmware)
    monkeypatch.setattr(sr_mod.serial, "Serial", SilentSerial)
    with pytest.raises(ConnectionError):
        sr_mod.SerialRig("COM_FAKE", reset_delay=0.0, connect_timeout=0.3)


def test_serial_rig_stale_telemetry_reports_not_ok(monkeypatch):
    monkeypatch.setattr(sr_mod.serial, "Serial", BrieflyTalks)
    rig = sr_mod.SerialRig("COM_FAKE", reset_delay=0.0)
    try:
        assert rig.telemetry().ok  # connected on the first frames
        time.sleep(sr_mod.STALE_S + 0.2)  # then the board fell silent
        assert rig.telemetry().ok is False  # failsafe signal to the mission
    finally:
        rig.close()


def test_serial_rig_bad_port_raises_connection_error(monkeypatch):
    def boom(*a, **k):
        raise sr_mod.serial.SerialException("no such port")

    monkeypatch.setattr(sr_mod.serial, "Serial", boom)
    with pytest.raises(ConnectionError):
        sr_mod.SerialRig("COM_NOPE", reset_delay=0.0)


def test_serial_rig_close_sends_safe_command(monkeypatch):
    monkeypatch.setattr(sr_mod.serial, "Serial", FakeSerial)
    rig = sr_mod.SerialRig("COM_FAKE", reset_delay=0.0)
    fake = rig._serial
    rig.close()
    assert fake.closed
    # the last thing written leaves the rig safe: pump 0, valve closed
    assert b"w=0.0" in fake.written[-1]
    assert b"v=0" in fake.written[-1]
