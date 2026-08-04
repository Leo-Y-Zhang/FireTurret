# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Browser interface: serves a live annotated video stream, status, and an
E-stop button over HTTP — Python standard library only, no extra dependencies.

Point a phone or laptop at the Raspberry Pi's address (e.g. http://raspberrypi.local:8000)
to watch the turret, read operator warnings, and hit E-stop remotely. It runs
over a real camera/video source or the built-in simulator, so it demos without
hardware.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ..config import TritonConfig
from ..control.advisories import Advisory, most_urgent
from ..rig.interface import NullRig, RigCommand, RigTelemetry, TurretRig

_INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Triton</title><style>
body{margin:0;background:#10151c;color:#e8edf4;font-family:system-ui,sans-serif;text-align:center}
h1{letter-spacing:.3em;font-weight:800;margin:.5rem}
#warn{display:none;padding:.6rem;font-weight:700;color:#fff}
img{max-width:100%;height:auto;border-top:1px solid #253141;border-bottom:1px solid #253141}
#s{color:#8fa0b5;font-size:.9rem;margin:.4rem}
button{background:#c0392b;color:#fff;border:0;border-radius:8px;padding:.9rem 2rem;
font-size:1.1rem;font-weight:700;margin:.6rem;cursor:pointer}
</style></head><body>
<h1>TRITON</h1><div id=warn></div>
<img src="/stream" alt="live view">
<div id=s>connecting...</div>
<button onclick="fetch('/estop',{method:'POST'})">E-STOP (toggle)</button>
<script>
async function poll(){try{let r=await fetch('/status');let d=await r.json();
let w=document.getElementById('warn');
if(d.warning){w.style.display='block';w.style.background=d.warning.severity=='CRITICAL'?'#c0392b':'#c47f17';
w.textContent='⚠ '+d.warning.message+' → '+d.warning.action;}else{w.style.display='none';}
document.getElementById('s').textContent='state: '+d.state+(d.estop?'  |  E-STOP':'')+
'  |  pan '+d.pan.toFixed(0)+'° tilt '+d.tilt.toFixed(0)+'° pump '+d.pump.toFixed(0)+'%';
}catch(e){}setTimeout(poll,400);}poll();
</script></body></html>""".encode()


class WebState:
    """Thread-safe holder for the latest annotated frame + status, shared
    between the pipeline worker and the HTTP handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg = b""
        self._status: dict = {"state": "-", "pan": 0, "tilt": 0, "pump": 0, "estop": False, "warning": None}
        self.estop = False
        self.stop = False

    def update(self, jpeg: bytes, status: dict) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._status = status

    def snapshot(self) -> tuple[bytes, dict]:
        with self._lock:
            return self._jpeg, dict(self._status)


def status_dict(state: str, cmd: RigCommand, tel: RigTelemetry,
                advisories: list[Advisory], estop: bool) -> dict:
    top = most_urgent(advisories)
    return {
        "state": state,
        "pan": tel.pan_deg, "tilt": tel.tilt_deg, "pump": tel.pump_pct,
        "valve": cmd.valve, "estop": estop,
        "warning": None if top is None
        else {"kind": top.kind, "severity": top.severity, "message": top.message, "action": top.action},
    }


def _make_handler(state: WebState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default logging
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") == "/estop":
                state.estop = not state.estop
                self._send(200, "application/json", json.dumps({"estop": state.estop}).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _INDEX_HTML)
            elif path == "/status":
                _, status = state.snapshot()
                self._send(200, "application/json", json.dumps(status).encode())
            elif path == "/stream":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not state.stop:
                    jpeg, _ = state.snapshot()
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def make_server(
    state: WebState,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    acknowledge_exposure: bool = False,
) -> ThreadingHTTPServer:
    """Serve the operator console.

    Defaults to LOOPBACK. This console has no authentication: it streams the
    camera and exposes ``POST /estop``, which latches the mission into SAFE and
    disarms water with no rearm path from the browser. Bound to a routable
    address that is remote, anonymous, irreversible denial of a fire-suppression
    device — so leaving the machine is a deliberate act the caller must
    acknowledge, not a default.
    """
    if host not in _LOOPBACK and not acknowledge_exposure:
        raise ValueError(
            f"refusing to bind the unauthenticated console to {host!r}: it exposes a "
            "camera stream and a remote E-stop with no authentication. Pass "
            "acknowledge_exposure=True (CLI: --expose) to accept that risk."
        )
    return ThreadingHTTPServer((host, port), _make_handler(state))


def encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else b""


def apply_estop(tel: RigTelemetry, estop: bool) -> RigTelemetry:
    """Inject a web-triggered E-stop into telemetry so the mission goes SAFE."""
    if not estop:
        return tel
    return replace(tel, estop=True)


def rig_command_for_hardware(
    command: RigCommand, estop: bool, water_enabled: bool
) -> RigCommand:
    """The command actually sent to the rig on the real-hardware path. An engaged
    E-stop OR dry-aim (water not armed) forces pump/valve OFF — mirroring the
    `run` command's two-layer dry-aim default so the web console can never spray
    on first bring-up unless the operator explicitly armed water."""
    if estop:
        return RigCommand(command.pan_deg, command.tilt_deg, 0.0, False, warn=command.warn)
    if not water_enabled:
        return replace(command, pump_pct=0.0, valve=False)
    return command


def serve(
    cfg: TritonConfig,
    source: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    rig: TurretRig | None = None,
    scenario=None,
    water_enabled: bool = False,
    acknowledge_exposure: bool = False,
) -> None:
    """Run the pipeline over `source` ('sim' or a camera index / video path) and
    serve the live interface. Blocks until interrupted.

    On real hardware (a non-sim source), water stays OFF (dry-aim) unless
    ``water_enabled`` is True — the same safe default as ``triton run``. The
    built-in ``sim`` source always sprays so the demo works with no hardware.

    Binds to LOOPBACK by default; reaching a non-loopback address requires
    ``acknowledge_exposure`` (CLI ``--expose``) because the console is
    unauthenticated and its E-stop is irreversible from the browser."""
    from ..app import Pipeline
    from ..ui.overlay import draw_overlay

    state = WebState()
    server = make_server(state, host, port, acknowledge_exposure=acknowledge_exposure)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Triton web interface on http://{host}:{port}  (Ctrl-C to stop)")

    # arm the control layer for the sim demo, or on hardware only when the
    # operator armed water; disarmed keeps the mission from ever commanding water
    pipeline = Pipeline(cfg, armed=(source == "sim") or water_enabled)
    rig = rig if rig is not None else NullRig()
    dt = 1 / 30
    try:
        if source == "sim":
            from ..rig.sim_rig import SimRig, SimScenario
            sim = SimRig(cfg, scenario or SimScenario(), seed=7)
            while not state.stop:
                sim.step(dt)
                frame = sim.render()
                tel = apply_estop(sim.telemetry(), state.estop)
                result = pipeline.tick(frame, dt, tel)
                sim.command(result.command)
                shown = draw_overlay(frame, cfg, result.tracks, result.target, result.impact,
                                     pipeline.mission.debug, result.command, tel, pipeline.mission.advisories)
                state.update(encode_jpeg(shown),
                             status_dict(pipeline.mission.state, result.command, tel,
                                         pipeline.mission.advisories, state.estop))
                time.sleep(dt)
        else:
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not cap.isOpened():
                raise RuntimeError(f"could not open camera/video source: {source!r}")
            while not state.stop:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (cfg.camera.width, cfg.camera.height):
                    frame = cv2.resize(frame, (cfg.camera.width, cfg.camera.height))
                tel = apply_estop(rig.telemetry(), state.estop)
                result = pipeline.tick(frame, dt, tel)
                # dry-aim by default on real hardware (belt-and-suspenders on top
                # of the disarmed control layer); E-stop also forces water off
                cmd = rig_command_for_hardware(result.command, state.estop, water_enabled)
                rig.command(cmd)
                shown = draw_overlay(frame, cfg, result.tracks, result.target, result.impact,
                                     pipeline.mission.debug, cmd, tel, pipeline.mission.advisories)
                state.update(encode_jpeg(shown),
                             status_dict(pipeline.mission.state, cmd, tel,
                                         pipeline.mission.advisories, state.estop))
            cap.release()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop = True
        server.shutdown()
        rig.close()
