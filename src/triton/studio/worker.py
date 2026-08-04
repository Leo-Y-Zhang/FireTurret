# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SimWorker: runs `simcore.drive` on a worker thread and streams frozen telemetry
batches + latest frames to the GUI via Qt signals. It NEVER touches widgets; all
communication is via signals (auto-queued across the thread boundary).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

from PySide6.QtCore import QObject, Signal, Slot

from ..config import TritonConfig
from ..rig.sim_rig import SimScenario
from ..simcore import SimStep, drive
from .throttle import EmissionThrottle


class SimWorker(QObject):
    progress = Signal(float)     # 0..1
    telemetry = Signal(object)   # list[TelemetrySample] (lossless batch)
    frame = Signal(object)       # np.ndarray (latest raw frame)
    finished = Signal(object)    # SimReport
    failed = Signal(str)

    def __init__(self, cfg: TritonConfig, scenario: SimScenario | None = None, *,
                 seed: int = 7, max_frames: int = 6000, record_telemetry: bool = True,
                 steps: Iterator[SimStep] | None = None,
                 now: Callable[[], float] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._scenario = scenario
        self._seed = seed
        self._max_frames = max_frames
        self._record = record_telemetry
        self._steps = steps
        self._now = now or time.monotonic
        self._cancel = False
        self._throttle: EmissionThrottle | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation (read between ticks by `drive`)."""
        self._cancel = True

    @Slot()
    def run(self) -> None:
        try:
            self._throttle = EmissionThrottle(self._now)
            report = drive(
                self._cfg, self._scenario, seed=self._seed, max_frames=self._max_frames,
                record_telemetry=self._record, on_step=self._on_step,
                should_stop=lambda: self._cancel, steps=self._steps,
            )
            batch, frame = self._throttle.flush()
            if batch:
                self.telemetry.emit(batch)
            if frame is not None:
                self.frame.emit(frame)
            self.finished.emit(report)
        except Exception as exc:  # surfaced to the GUI, never a bare crash
            self.failed.emit(repr(exc))

    def _on_step(self, step: SimStep) -> None:
        batch = self._throttle.offer_sample(step.sample)
        if batch is not None:
            self.telemetry.emit(batch)
        # offer_frame is latest-wins and returns the CURRENT frame when it fires,
        # so overlaying with this step's fields stays in sync.
        frame = self._throttle.offer_frame(step.frame)
        if frame is not None:
            self.frame.emit(self._overlay(step, frame))
        if self._max_frames:
            self.progress.emit(min(1.0, (step.frame_index + 1) / self._max_frames))

    def _overlay(self, step: SimStep, frame):
        # draw_overlay is pure cv2 drawing on a copy — safe off the main thread.
        # Stub steps (tests) carry result=None; pass the raw frame through.
        if step.result is None:
            return frame
        from ..ui.overlay import draw_overlay

        return draw_overlay(
            frame, self._cfg, step.result.tracks, step.result.target, step.result.impact,
            step.mission_debug, step.result.command, step.rig_telemetry,
            list(step.advisories),
        )
