# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Emission throttle for the SimWorker.

The simulation runs at full tick rate on the worker thread; this coalesces how
OFTEN data is emitted to the GUI without dropping any telemetry:

- telemetry is **lossless** — samples accumulate and are flushed as a batch at
  the target rate, so concatenating every emitted batch (plus `flush()`)
  reproduces the full per-tick stream in order;
- frames are **latest-wins** — only the most recent frame is kept between
  emissions (older frames are dropped), because a live view only needs the newest.

The clock is injected (`now`) so tests can drive virtual time deterministically.
"""
from __future__ import annotations

from collections.abc import Callable


class EmissionThrottle:
    def __init__(self, now: Callable[[], float], telemetry_hz: float = 25.0,
                 frame_hz: float = 15.0) -> None:
        self._now = now
        self._tel_dt = 1.0 / telemetry_hz
        self._frame_dt = 1.0 / frame_hz
        self._buf: list = []
        self._pending_frame = None
        self._last_tel: float | None = None
        self._last_frame: float | None = None

    def offer_sample(self, sample) -> list | None:
        """Buffer a sample; return the accumulated batch when the interval elapses."""
        self._buf.append(sample)
        t = self._now()
        if self._last_tel is None:
            self._last_tel = t
        if t - self._last_tel >= self._tel_dt:
            batch = self._buf
            self._buf = []
            self._last_tel = t
            return batch
        return None

    def offer_frame(self, frame) -> object | None:
        """Keep the latest frame; return it when the interval elapses (else drop)."""
        self._pending_frame = frame
        t = self._now()
        if self._last_frame is None:
            self._last_frame = t
        if t - self._last_frame >= self._frame_dt:
            frame = self._pending_frame
            self._pending_frame = None
            self._last_frame = t
            return frame
        return None

    def flush(self) -> tuple[list, object | None]:
        """Return any buffered telemetry batch and the latest pending frame."""
        batch = self._buf
        self._buf = []
        frame = self._pending_frame
        self._pending_frame = None
        return batch, frame
