# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Advisory scene understanding — strictly deny-only, off by default.

A vision-language model can notice things the classical stack cannot: that a fire
is electrical, that there is a gas cylinder beside it, that the burning liquid is
cooking oil. Those are genuinely useful, and for several of them **water is the
wrong agent entirely** — an electrical fire, a grease fire, a chemical fire. A
turret that sprays them makes things worse.

## What can and cannot be tested

The model's *judgement* cannot be tested. No assertion in this repository can
establish that a language model correctly recognises a grease fire, and none
pretends to.

What CAN be tested exhaustively is the **plumbing and the conservative-only
property**: no output of any provider, in any combination, can ever turn a denial
into an allowance. That claim is finite, and it is proven over the boolean
product with a stub — exactly as SP9 proves the veto without a detector.

So the model gets to say "stop". It never gets to say "go".
"""

from .provider import (
    HazardAssessment,
    LocalSceneProvider,
    NullSceneProvider,
    RemoteSceneProvider,
    SceneProvider,
    SceneVeto,
    WrongAgent,
)

__all__ = [
    "HazardAssessment",
    "LocalSceneProvider",
    "NullSceneProvider",
    "RemoteSceneProvider",
    "SceneProvider",
    "SceneVeto",
    "WrongAgent",
]
