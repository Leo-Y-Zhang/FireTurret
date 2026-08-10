# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Cover the browser interface: status payload, E-stop injection, and that the
HTTP server actually serves the page, live-stream headers, and status JSON."""

import json
import threading
import urllib.request

from fireturret.control.advisories import CRITICAL, Advisory
from fireturret.rig.interface import RigCommand, RigTelemetry
from fireturret.ui.webserver import (
    WebState,
    apply_estop,
    encode_jpeg,
    make_server,
    rig_command_for_hardware,
    status_dict,
)


def test_status_dict_shapes_payload() -> None:
    cmd = RigCommand(10.0, 25.0, 55.0, True)
    tel = RigTelemetry(10.0, 25.0, 55.0, True, estop=False, ok=True)
    adv = [Advisory("TOO_FAR", CRITICAL, "Fire out of reach", "Move closer")]
    d = status_dict("HOLD", cmd, tel, adv, estop=False)
    assert d["state"] == "HOLD"
    assert d["warning"]["kind"] == "TOO_FAR"
    assert d["warning"]["severity"] == CRITICAL
    empty = status_dict("SEARCH", cmd, tel, [], estop=False)
    assert empty["warning"] is None


def test_hardware_command_is_dry_by_default() -> None:
    """On the real-hardware path, water is OFF unless explicitly armed — the
    web console must match `fireturret run`'s dry-aim-first safety default."""
    cmd = RigCommand(10.0, 25.0, 55.0, True, warn=True)
    # not armed: pump/valve forced off, but aim + warn still pass through
    dry = rig_command_for_hardware(cmd, estop=False, water_enabled=False)
    assert dry.pump_pct == 0.0 and dry.valve is False
    assert dry.pan_deg == 10.0 and dry.tilt_deg == 25.0 and dry.warn is True
    # armed: the full command passes through so the turret can actually spray
    wet = rig_command_for_hardware(cmd, estop=False, water_enabled=True)
    assert wet.pump_pct == 55.0 and wet.valve is True
    # E-stop kills water regardless of arming
    stopped = rig_command_for_hardware(cmd, estop=True, water_enabled=True)
    assert stopped.pump_pct == 0.0 and stopped.valve is False


def test_apply_estop_injects_estop() -> None:
    tel = RigTelemetry(0, 0, 0, False, estop=False, ok=True)
    assert apply_estop(tel, False).estop is False
    assert apply_estop(tel, True).estop is True


def test_server_serves_page_and_status() -> None:
    import numpy as np

    state = WebState()
    state.update(encode_jpeg(np.zeros((60, 80, 3), np.uint8)),
                 {"state": "SEARCH", "pan": 1.0, "tilt": 2.0, "pump": 0.0, "estop": False, "warning": None})
    server = make_server(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/", timeout=3) as r:
            assert r.status == 200
            assert b"FIRETURRET" in r.read()
        with urllib.request.urlopen(base + "/status", timeout=3) as r:
            payload = json.loads(r.read())
            assert payload["state"] == "SEARCH"
        # E-stop toggles via POST
        req = urllib.request.Request(base + "/estop", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            assert json.loads(r.read())["estop"] is True
    finally:
        state.stop = True
        server.shutdown()
