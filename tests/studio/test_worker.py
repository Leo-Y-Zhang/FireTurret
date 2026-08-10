# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 7: SimWorker streaming, failure, and deterministic threaded cancel."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QThread

from fireturret.analysis import TelemetrySample
from fireturret.app import SimReport
from fireturret.config import DEFAULT_CONFIG
from fireturret.simcore import SimStep
from fireturret.studio.worker import SimWorker


def _make_step(i: int, state: str = "SEARCH", extinguished: bool = False) -> SimStep:
    sample = TelemetrySample(
        t=i / 30.0, state=state, pan_cmd=0.0, pan_act=0.0, tilt=8.0, pump=0.0,
        valve=False, range_est_m=0.0, az_err_deg=0.0, range_err_px=0.0,
        fire_intensity=1.0, water_l=0.0, splash_miss_m=0.0, warning="",
    )
    return SimStep(
        frame=np.zeros((4, 4, 3), np.uint8), result=None, sample=sample, advisories=(),
        mission_debug=None, rig_telemetry=None, primary_intensity=1.0,
        fire_intensities=(1.0,), miss_recorded=False, extinguished=extinguished,
        water_used_l=0.0, frame_index=i, sim_seconds=i / 30.0,
    )


def test_worker_emits_lossless_telemetry_and_finished_sync(qapp):
    # frozen clock -> throttle flushes everything at the end (deterministic, lossless)
    w = SimWorker(DEFAULT_CONFIG, seed=7, max_frames=120, now=lambda: 0.0)
    tel: list = []
    reports: list = []
    w.telemetry.connect(tel.append)   # same thread => direct connection, safe
    w.finished.connect(reports.append)
    w.run()

    assert len(reports) == 1
    assert isinstance(reports[0], SimReport)
    flat = [s for b in tel for s in b]
    assert len(flat) == reports[0].frames   # one emitted sample per folded tick, none lost
    assert flat == reports[0].telemetry     # exactly the samples the report retained


def test_worker_failed_signal_on_exception(qapp):
    def bad():
        yield _make_step(0)
        raise RuntimeError("boom")

    w = SimWorker(DEFAULT_CONFIG, steps=bad(), now=lambda: 0.0)
    errs: list = []
    w.failed.connect(errs.append)
    w.run()
    assert errs and "boom" in errs[0]


def test_overlay_passthrough_when_no_result(qapp):
    w = SimWorker(DEFAULT_CONFIG)
    step = _make_step(0)  # stub step: result is None
    assert w._overlay(step, step.frame) is step.frame


def test_overlay_draws_for_real_step(qapp):
    from fireturret.simcore import simulate

    w = SimWorker(DEFAULT_CONFIG)
    step = next(simulate(DEFAULT_CONFIG, seed=7, max_frames=1))
    out = w._overlay(step, step.frame)
    assert out.shape == step.frame.shape
    assert out is not step.frame  # draw_overlay returns a copy


def test_worker_cancel_on_thread_is_deterministic(qapp, qtbot):
    started = threading.Event()
    gate = threading.Event()

    def gen():
        yield _make_step(0)
        started.set()
        gate.wait(3.0)          # block until the test releases, after cancel()
        for i in range(1, 300):
            yield _make_step(i)

    w = SimWorker(DEFAULT_CONFIG, steps=gen(), now=lambda: 0.0)
    th = QThread()
    w.moveToThread(th)
    th.started.connect(w.run)
    w.finished.connect(th.quit)
    try:
        with qtbot.waitSignal(w.finished, timeout=5000) as blocker:
            th.start()
            assert started.wait(3.0)   # worker folded step 0 and is now blocked
            w.cancel()
            gate.set()
        report = blocker.args[0]
        assert report.frames <= 1      # broke at the first should_stop after cancel
    finally:
        w.cancel()
        gate.set()
        th.quit()
        th.wait(3000)
