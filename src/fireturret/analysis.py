# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Mission telemetry logging, data export, and analysis reports.

A run can record a per-frame telemetry trace; from it, `write_csv` exports the
raw data and `write_report` renders a multi-panel engineering figure (state
timeline, aim-error convergence, range tracking against the jet's reachable
envelope, pump pressure, and cumulative water). matplotlib is required only for
the report and is an optional dependency (`pip install fireturret[analysis]`).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class TelemetrySample:
    t: float  # sim seconds
    state: str
    pan_cmd: float
    pan_act: float
    tilt: float
    pump: float
    valve: bool
    range_est_m: float  # controller's estimated target range (0 if none)
    az_err_deg: float  # azimuth error (0 if none)
    range_err_px: float  # splash-vs-fire row error (0 if none)
    fire_intensity: float  # total fire intensity across the scene
    water_l: float  # cumulative water used
    splash_miss_m: float  # splash-to-nearest-fire distance (0 if not spraying)
    warning: str  # most-urgent advisory kind, or ""


def write_csv(samples: list[TelemetrySample], path: str) -> None:
    cols = [f.name for f in fields(TelemetrySample)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))


def read_csv(path: str) -> list[TelemetrySample]:
    """Reconstruct a telemetry trace written by ``write_csv`` (for run replay)."""
    flds = fields(TelemetrySample)
    out: list[TelemetrySample] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            kwargs = {}
            for f in flds:
                value = row[f.name]
                if f.type == "bool":
                    kwargs[f.name] = value == "True"
                elif f.type == "str":
                    kwargs[f.name] = value
                else:
                    kwargs[f.name] = float(value)
            out.append(TelemetrySample(**kwargs))
    return out


# ordered mission phases for the state-timeline lane
_STATES = ["SEARCH", "ACQUIRE", "RANGE", "ALIGN", "SUPPRESS", "CONFIRM", "HOLD", "SAFE"]


def write_report(
    samples: list[TelemetrySample],
    path: str,
    min_reach_m: float,
    max_reach_m: float,
    title: str = "FireTurret mission",
) -> None:
    """Render a multi-panel analysis figure to `path` (PNG). Requires matplotlib."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    if not samples:
        raise ValueError("no telemetry samples to plot")

    t = [s.t for s in samples]
    ink, grid, accent = "#e8edf4", "#253141", "#4cc3e8"
    water_c, heat_c, ok_c = "#4cc3e8", "#ff9f43", "#58c470"

    plt.rcParams.update({
        "figure.facecolor": "#10151c", "axes.facecolor": "#141b24",
        "axes.edgecolor": grid, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink, "ytick.color": ink, "grid.color": grid,
        "font.size": 9,
    })
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(title, color=ink, fontsize=13, fontweight="bold")

    # 1) state timeline
    ax = axes[0]
    idx = {s: i for i, s in enumerate(_STATES)}
    ax.step(t, [idx.get(s.state, -1) for s in samples], where="post", color=accent, lw=1.5)
    ax.set_yticks(range(len(_STATES)))
    ax.set_yticklabels(_STATES, fontsize=7)
    ax.set_ylabel("mission state")
    ax.set_ylim(-0.5, len(_STATES) - 0.5)

    # 2) aim errors
    ax = axes[1]
    ax.axhline(0, color=grid, lw=1)
    ax.plot(t, [s.az_err_deg for s in samples], color=accent, lw=1, label="azimuth error (deg)")
    ax.plot(t, [s.range_err_px / 20.0 for s in samples], color=heat_c, lw=1, label="range error (px/20)")
    ax.set_ylabel("aim error")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.2)

    # 3) range tracking vs reachable envelope
    ax = axes[2]
    ax.axhspan(min_reach_m, max_reach_m, color=ok_c, alpha=0.12, label="reachable envelope")
    ax.plot(t, [s.range_est_m for s in samples], color=ink, lw=1.2, label="target range est (m)")
    ax.plot(t, [s.splash_miss_m for s in samples], color=water_c, lw=1, label="splash miss (m)")
    ax.set_ylabel("range / miss (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.2)

    # 4) pump pressure and valve
    ax = axes[3]
    ax.plot(t, [s.pump for s in samples], color=water_c, lw=1.2, label="pump (%)")
    ax.fill_between(t, 0, [100 if s.valve else 0 for s in samples], color=water_c,
                    alpha=0.12, step="post", label="valve open")
    ax.set_ylabel("pump / valve")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.2)

    # 5) fire knockdown and water used
    ax = axes[4]
    ax.plot(t, [s.fire_intensity for s in samples], color=heat_c, lw=1.5, label="total fire intensity")
    ax.set_ylabel("fire intensity", color=heat_c)
    ax2 = ax.twinx()
    ax2.plot(t, [s.water_l for s in samples], color=water_c, lw=1.2, label="water used (L)")
    ax2.set_ylabel("water (L)", color=water_c)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.2)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.2)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)
