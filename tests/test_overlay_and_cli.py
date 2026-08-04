# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import numpy as np
import pytest

from triton.__main__ import SCENARIOS, main
from triton.config import DEFAULT_CONFIG
from triton.control.mission import MissionDebug
from triton.rig.interface import RigCommand, RigTelemetry
from triton.ui.overlay import draw_overlay


def _frame() -> np.ndarray:
    return np.zeros((DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3), dtype=np.uint8)


@pytest.mark.parametrize("state", ["SEARCH", "ALIGN", "SUPPRESS", "HOLD", "SAFE"])
def test_overlay_renders_without_error(state: str) -> None:
    debug = MissionDebug(state=state, target_range_m=8.0, azimuth_error_deg=-0.3,
                         range_error_px=5.0, spray_elapsed_s=2.0,
                         note="target out of reach" if state == "HOLD" else "")
    cmd = RigCommand(12.0, 22.0, 55.0, state == "SUPPRESS")
    tel = RigTelemetry(12.0, 22.0, 55.0, state == "SUPPRESS", estop=(state == "SAFE"), ok=True)
    out = draw_overlay(_frame(), DEFAULT_CONFIG, [], None, None, debug, cmd, tel)
    assert out.shape == _frame().shape
    assert out.dtype == np.uint8


def test_cli_scenarios_are_all_valid() -> None:
    assert set(SCENARIOS) >= {"default", "close", "offset", "far", "unreachable", "multi"}


def test_cli_sim_runs_headless() -> None:
    rc = main(["sim", "--scenario", "close", "--headless", "--frames", "300"])
    assert rc in (0, 1)  # 0 extinguished, 1 not (both are clean exits)
