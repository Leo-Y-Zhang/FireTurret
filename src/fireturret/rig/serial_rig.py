# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Serial driver for the real turret: pyserial link to the microcontroller
running firmware/turret_firmware. A background reader keeps the latest
telemetry; telemetry older than the staleness window reports ok=False, which
the mission treats as a failsafe condition.
"""

from __future__ import annotations

import threading
import time

import serial

from .interface import RigCommand, RigTelemetry
from .protocol import encode_command, parse_telemetry

STALE_S = 0.6


def list_serial_ports() -> list[str]:
    """Device names of the serial ports currently present (for error hints)."""
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover
        return []
    return [p.device for p in list_ports.comports()]


class SerialRig:
    def __init__(self, port: str, baud: int = 115200, *,
                 connect_timeout: float = 6.0, reset_delay: float = 1.5) -> None:
        try:
            self._serial = serial.Serial(port, baudrate=baud, timeout=0.2)
        except (serial.SerialException, OSError) as exc:
            ports = list_serial_ports()
            hint = f" Available ports: {', '.join(ports)}." if ports else " No serial ports detected."
            raise ConnectionError(f"could not open {port}: {exc}.{hint}") from exc
        self._lock = threading.Lock()
        self._telemetry: RigTelemetry | None = None
        self._telemetry_at = 0.0
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        time.sleep(reset_delay)  # many boards reset on serial open

        # Confirm the board is actually TALKING — a port that opens but is the
        # wrong device / baud / has no firmware would otherwise look "connected"
        # and then silently never actuate.
        deadline = time.monotonic() + connect_timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._telemetry is not None:
                    return
            time.sleep(0.05)
        self.close()
        raise ConnectionError(
            f"opened {port} but received no telemetry within {connect_timeout:g}s — "
            f"check the board is flashed with turret_firmware, the baud is {baud}, "
            "and the TX/RX wiring."
        )

    def _read_loop(self) -> None:
        buffer = b""
        while self._running:
            try:
                chunk = self._serial.read(64)
            except (serial.SerialException, OSError):
                break
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                telemetry = parse_telemetry(line.decode("ascii", errors="replace"))
                if telemetry is not None:
                    with self._lock:
                        self._telemetry = telemetry
                        self._telemetry_at = time.monotonic()

    def command(self, cmd: RigCommand) -> None:
        try:
            self._serial.write(encode_command(cmd).encode("ascii"))
        except (serial.SerialException, OSError):
            # Link lost mid-mission (e.g. USB jostled, board brown-out/re-enumerate).
            # Tear it down so telemetry() reports ok=False immediately and the
            # mission degrades to SAFE next tick, instead of an uncaught traceback
            # aborting the run. Mirrors the read loop's graceful handling.
            self._running = False
            with self._lock:
                self._telemetry = None

    def telemetry(self) -> RigTelemetry:
        with self._lock:
            telemetry = self._telemetry
            age = time.monotonic() - self._telemetry_at
        if telemetry is None or age > STALE_S:
            return RigTelemetry(0.0, 0.0, 0.0, False, estop=False, ok=False)
        return telemetry

    def close(self) -> None:
        self._running = False
        try:
            # leave the rig safe regardless of mission state
            self.command(RigCommand(0.0, 20.0, 0.0, False))
        except (serial.SerialException, OSError):
            pass
        self._serial.close()
