# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Platform pose, and an explicit ladder for how much to trust it.

## Why quality is a ladder and not a boolean

"Do we know where the base is pointing" has more than two useful answers, and
collapsing them loses the one that matters most. A pose can be:

    PERFECT     a static platform, known by construction — the fixed turret
    GOOD        a converged filter with recent, consistent measurements
    DEGRADED    usable for tracking, NOT for committing water
    LOST        no usable attitude at all

DEGRADED is the rung that earns the ladder. A boolean forces it into either
"fine" (and the turret sprays on a corrupted pose) or "broken" (and the turret
stops working every time the estimate wobbles). Neither is right: with a degraded
pose the machine should keep *looking* — tracking, searching, staying pointed at
the fire — and simply not commit water. That is a different behaviour from both
alternatives, and it is the useful one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PoseQuality(IntEnum):
    """Ordered worst to best, so comparisons read naturally:
    `if quality < PoseQuality.GOOD: ...`"""

    LOST = 0
    DEGRADED = 1
    GOOD = 2
    PERFECT = 3

    @property
    def may_commit_water(self) -> bool:
        """Water needs a pose you can aim with. DEGRADED explicitly does not
        qualify — that is the whole reason the rung exists."""
        return self >= PoseQuality.GOOD


@dataclass(frozen=True)
class PlatformState:
    """Where the base is, how fast it is turning, and how much to believe it."""

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate_dps: float = 0.0
    pitch_rate_dps: float = 0.0
    yaw_rate_dps: float = 0.0
    translation_m: float = 0.0  # accumulated ground distance travelled
    quality: PoseQuality = PoseQuality.PERFECT

    @property
    def is_static(self) -> bool:
        """The fixed-turret case: no attitude, no rates, perfect knowledge.

        Checked as an exact zero rather than with a tolerance, deliberately. This
        predicate decides whether the code takes the historical path, and "very
        nearly static" must NOT silently take it — that would make the
        fixed-turret goldens depend on a threshold.
        """
        return (
            self.roll_deg == 0.0
            and self.pitch_deg == 0.0
            and self.yaw_deg == 0.0
            and self.roll_rate_dps == 0.0
            and self.pitch_rate_dps == 0.0
            and self.yaw_rate_dps == 0.0
            and self.quality is PoseQuality.PERFECT
        )

    @property
    def total_rate_dps(self) -> float:
        """The single ω_total the stabiliser's rate budget is derived from.

        Written once, here, so the pan and tilt rate requirements cannot drift
        apart — and so a test can check them against the same expression rather
        than against two independently-typed constants.
        """
        return (
            self.roll_rate_dps**2 + self.pitch_rate_dps**2 + self.yaw_rate_dps**2
        ) ** 0.5


STATIC = PlatformState()
