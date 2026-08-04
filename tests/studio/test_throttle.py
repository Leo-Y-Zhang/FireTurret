# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 7: the emission throttle is lossless (telemetry) and latest-wins (frames)."""
from __future__ import annotations

from triton.studio.throttle import EmissionThrottle


def test_telemetry_is_lossless_and_bounded():
    t = {"v": 0.0}
    thr = EmissionThrottle(now=lambda: t["v"], telemetry_hz=25.0, frame_hz=15.0)
    emitted: list[list] = []
    for i in range(100):
        t["v"] += 0.001  # 1 ms per sample -> 100 samples over 0.1 s
        batch = thr.offer_sample(i)
        if batch is not None:
            emitted.append(batch)
    rem, _ = thr.flush()
    if rem:
        emitted.append(rem)

    flat = [x for b in emitted for x in b]
    assert flat == list(range(100))       # every sample, in order (lossless)
    assert 1 <= len(emitted) <= 5         # coalesced to ~25 Hz, not 100 emissions


def test_frames_are_latest_wins():
    t = {"v": 0.0}
    thr = EmissionThrottle(now=lambda: t["v"], telemetry_hz=25.0, frame_hz=15.0)
    emitted = []
    for i in range(100):
        t["v"] += 0.001
        f = thr.offer_frame(i)
        if f is not None:
            emitted.append(f)

    assert len(emitted) < 100             # frames are dropped, not queued
    assert emitted == sorted(emitted)     # each emitted frame is the newest so far


def test_frozen_clock_emits_only_on_flush():
    thr = EmissionThrottle(now=lambda: 0.0)
    for i in range(50):
        assert thr.offer_sample(i) is None  # never crosses the interval
    batch, _ = thr.flush()
    assert batch == list(range(50))
