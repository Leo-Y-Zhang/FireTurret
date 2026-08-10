# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Operator overlay: everything the pipeline knows, drawn on the frame —
detections, the confirmed target, splash observation, mission state, aim
solution, telemetry, and a side-view of the predicted water arc.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import ballistics
from ..config import FireTurretConfig
from ..control.advisories import CRITICAL, Advisory, most_urgent
from ..control.mission import MissionDebug
from ..rig.interface import RigCommand, RigTelemetry
from ..vision.impact import ImpactObservation
from ..vision.tracker import Track

INK = (235, 238, 240)
MUTED = (150, 160, 165)
FIRE = (60, 110, 255)
TARGET = (40, 60, 255)
WATER = (250, 210, 120)
OK = (120, 210, 130)
WARN = (60, 170, 250)

STATE_COLOURS = {
    "SEARCH": (200, 190, 120),
    "ACQUIRE": WARN,
    "RANGE": WARN,
    "ALIGN": WARN,
    "SUPPRESS": WATER,
    "CONFIRM": OK,
    "HOLD": (120, 120, 210),
    "SAFE": (80, 80, 230),
}


def draw_overlay(
    frame: np.ndarray,
    cfg: FireTurretConfig,
    tracks: list[Track],
    target: Track | None,
    impact: ImpactObservation | None,
    debug: MissionDebug,
    command: RigCommand,
    telemetry: RigTelemetry,
    advisories: list[Advisory] | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # detections and the confirmed target
    for track in tracks:
        blob = track.last_blob
        if blob is None:
            continue
        x, y, bw, bh = blob.bbox
        is_target = target is not None and track.track_id == target.track_id
        colour = TARGET if is_target else FIRE
        thickness = 2 if is_target else 1
        cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, thickness)
        label = f"FIRE #{track.track_id} {track.confidence:.2f}"
        cv2.putText(out, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        if is_target:
            gx, gy = int(track.cx), int(y + bh)
            cv2.drawMarker(out, (gx, gy), TARGET, cv2.MARKER_TILTED_CROSS, 16, 2)

    if impact is not None:
        cv2.circle(out, (int(impact.cx), int(impact.cy)), 10, WATER, 2)
        cv2.putText(out, "SPLASH", (int(impact.cx) + 12, int(impact.cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WATER, 1, cv2.LINE_AA)

    # boresight
    cv2.drawMarker(out, (w // 2, h // 2), MUTED, cv2.MARKER_CROSS, 14, 1)

    # state banner
    state_colour = STATE_COLOURS.get(debug.state, INK)
    cv2.rectangle(out, (0, 0), (w, 26), (20, 22, 25), -1)
    cv2.putText(out, f"FIRETURRET  |  {debug.state}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_colour, 2, cv2.LINE_AA)
    if debug.state == "SUPPRESS":
        cv2.putText(out, f"spray {debug.spray_elapsed_s:4.1f}s", (w - 130, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WATER, 1, cv2.LINE_AA)
    if debug.note:
        cv2.putText(out, debug.note.upper(), (w - 260, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 230), 1, cv2.LINE_AA)

    # telemetry block
    lines = [
        f"pan  cmd {command.pan_deg:7.2f}  act {telemetry.pan_deg:7.2f}",
        f"tilt cmd {command.tilt_deg:7.2f}  act {telemetry.tilt_deg:7.2f}",
        f"pump cmd {command.pump_pct:5.1f}%  act {telemetry.pump_pct:5.1f}%",
        f"valve {'OPEN' if command.valve else 'closed'}   estop {'!' if telemetry.estop else 'ok'}",
    ]
    if debug.target_range_m is not None:
        lines.append(f"range est {debug.target_range_m:5.1f} m")
    if debug.solution is not None:
        s = debug.solution
        lines.append(f"solution: {s.elevation_deg:4.1f} deg @ {s.pump_pct:3.0f}% -> {s.predicted_range_m:4.1f} m")
    if debug.azimuth_error_deg is not None:
        lines.append(f"az err {debug.azimuth_error_deg:+5.2f} deg")
    if debug.range_error_px is not None:
        lines.append(f"rng err {debug.range_error_px:+5.0f} px")
    # leave room for the advisory banner (drawn at y 30–76) when one is present
    y0 = 92 if most_urgent(advisories or []) is not None else 44
    for line in lines:
        cv2.putText(out, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
        y0 += 17

    _draw_arc_inset(out, cfg, debug, command)
    _draw_advisories(out, advisories or [])
    return out


def _draw_advisories(out: np.ndarray, advisories: list[Advisory]) -> None:
    """Prominent operator warning banner: what's wrong and what to do about it."""
    top = most_urgent(advisories)
    if top is None:
        return
    h, w = out.shape[:2]
    critical = top.severity == CRITICAL
    fill = (40, 40, 200) if critical else (30, 120, 210)  # red / amber (BGR)
    by = 30
    bh = 46
    cv2.rectangle(out, (0, by), (w, by + bh), fill, -1)
    label = f"{'!! ' if critical else '! '}{top.kind.replace('_', ' ')}"
    cv2.putText(out, label, (10, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, top.message, (140, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"-> {top.action}", (140, by + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if len(advisories) > 1:
        cv2.putText(out, f"(+{len(advisories) - 1} more)", (w - 110, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_arc_inset(
    out: np.ndarray, cfg: FireTurretConfig, debug: MissionDebug, command: RigCommand
) -> None:
    """Side view of the commanded water arc, bottom-left."""
    box_w, box_h = 230, 110
    h = out.shape[0]
    x0, y0 = 10, h - box_h - 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (20, 22, 25), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), MUTED, 1)
    cv2.putText(out, "arc (side view)", (x0 + 6, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)

    speed = ballistics.exit_velocity(command.pump_pct, cfg.jet)
    if speed < 1.0:
        return
    arc = ballistics.simulate_arc(speed, command.tilt_deg, cfg.jet)
    max_x = max(6.0, arc[-1][0] * 1.1)
    max_y = max(2.0, max(p[1] for p in arc) * 1.4)
    pts = []
    for px, py in arc:
        sx = x0 + 8 + int((box_w - 16) * (px / max_x))
        sy = y0 + box_h - 8 - int((box_h - 26) * (py / max_y))
        pts.append((sx, sy))
    cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, WATER, 2)
    # ground line + landing range label
    cv2.line(out, (x0 + 8, y0 + box_h - 8), (x0 + box_w - 8, y0 + box_h - 8), MUTED, 1)
    cv2.putText(out, f"{arc[-1][0]:4.1f} m", (pts[-1][0] - 34, y0 + box_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WATER, 1, cv2.LINE_AA)
    if debug.target_range_m is not None:
        tx = x0 + 8 + int((box_w - 16) * (min(debug.target_range_m, max_x) / max_x))
        cv2.drawMarker(out, (tx, y0 + box_h - 8), TARGET, cv2.MARKER_TRIANGLE_UP, 10, 2)
