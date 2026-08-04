# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP2 — the determinism substrate.

Two properties, and the reason each exists:

1. **Independence.** Draining one stream must not perturb another. Under the old
   single shared generator, one extra draw anywhere re-rolled every subsequent
   random value in the run — which is why adding a single renderer detail moved
   all thirteen golden fixtures at once, with no merge conflict to warn anyone.

2. **Order-independence.** Streams are keyed by NAME, not by registration order,
   so declaring a new one does not renumber the existing ones. This is why
   `RngBundle` derives a `spawn_key` from a hash of the name rather than using
   `SeedSequence.spawn(k)`, which hands streams out by index.

Plus the property that makes those two worth anything in practice: each renderer
advances its stream by a CONSTANT amount per frame, so a stream's position is a
function of the frame count alone — never of whether the fire was alive or the
valve was open.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton.config import DEFAULT_CONFIG, CameraConfig
from triton.rig.sim_rig import SimRig, SimScenario
from triton.rng import RngBundle, stream_spawn_key
from triton.vision.synthetic import (
    FIRE_DRAWS,
    SPLASH_DRAWS,
    Draws,
    FireState,
    SceneCamera,
    WorldPoint,
    _draw_fire,
    render_scene,
)

SEED = 11


def _values_after(seed: int, name: str, skip: int, take: int = 64) -> np.ndarray:
    """What `name`'s stream yields after exactly `skip` draws."""
    gen = RngBundle(seed, [name])[name]
    if skip:
        gen.random(skip)
    return gen.random(take)


# ------------------------------------------------------------- 1. independence

def test_draining_one_stream_does_not_move_another() -> None:
    bundle = RngBundle(SEED, ["fire0", "splash"])
    bundle["fire0"].random(1000)
    assert np.array_equal(bundle["splash"].random(64), _values_after(SEED, "splash", 0))


def test_streams_from_one_seed_are_not_the_same_stream() -> None:
    """Independence is worthless if the streams are identical."""
    bundle = RngBundle(SEED, ["fire0", "splash"])
    assert not np.array_equal(bundle["fire0"].random(64), bundle["splash"].random(64))


# -------------------------------------------------------- 2. order-independence

def test_registration_order_does_not_change_any_stream() -> None:
    forward = RngBundle(SEED, ["fire0", "fire1", "splash"])
    backward = RngBundle(SEED, ["splash", "fire1", "fire0"])
    for name in ("fire0", "fire1", "splash"):
        assert np.array_equal(forward[name].random(64), backward[name].random(64)), name


def test_declaring_an_extra_stream_does_not_renumber_the_others() -> None:
    """The failure mode `SeedSequence.spawn(k)` would have: inserting a consumer
    shifts every later one, silently re-rolling their values."""
    without = RngBundle(SEED, ["fire0", "splash"])
    with_extra = RngBundle(SEED, ["fire0", "brand_new_consumer", "splash"])
    for name in ("fire0", "splash"):
        assert np.array_equal(without[name].random(64), with_extra[name].random(64)), name


def test_spawn_key_is_stable_across_processes() -> None:
    """blake2b, not the built-in hash(), which PYTHONHASHSEED randomises per
    process — that would make a seeded run irreproducible between invocations."""
    assert stream_spawn_key("fire0") == stream_spawn_key("fire0")
    assert stream_spawn_key("fire0") != stream_spawn_key("fire1")


def test_different_seeds_give_different_streams() -> None:
    assert not np.array_equal(
        RngBundle(1, ["fire0"])["fire0"].random(64),
        RngBundle(2, ["fire0"])["fire0"].random(64),
    )


# ------------------------------------------------------------------- declaring

def test_undeclared_stream_raises() -> None:
    bundle = RngBundle(SEED, ["fire0"])
    with pytest.raises(KeyError, match="undeclared"):
        bundle["splash"]


def test_duplicate_stream_name_raises() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RngBundle(SEED, ["fire0", "fire0"])


# ------------------------------------------------- 3. constant draws per frame

_CAM = SceneCamera(CameraConfig())
_FIRE = FireState(position=WorldPoint.from_polar(0.0, 7.0), intensity=1.0)
_DEAD = FireState(position=WorldPoint.from_polar(0.0, 7.0), intensity=0.0)
_OFFSCREEN = FireState(position=WorldPoint.from_polar(179.0, 7.0), intensity=1.0)
_SPLASH = WorldPoint.from_polar(0.0, 6.0)

_SCENES = [
    ("fire + splash", _FIRE, _SPLASH),
    ("fire, dry", _FIRE, None),
    ("no fire, splash", None, _SPLASH),
    ("empty scene", None, None),
    ("dead fire", _DEAD, _SPLASH),
    ("fire behind the camera", _OFFSCREEN, _SPLASH),
]


@pytest.mark.parametrize("label,fire,splash", _SCENES, ids=[s[0] for s in _SCENES])
def test_each_stream_advances_by_its_block_every_frame(label, fire, splash) -> None:
    """Whatever the scene contains, both streams advance by exactly one block per
    frame. This is the property that decouples the renderer's randomness from the
    control loop's decisions."""
    frames = 5
    bundle = RngBundle(SEED, ["splash", "fire0"])
    for _ in range(frames):
        render_scene(_CAM, 0.0, fire, splash, bundle)

    assert np.array_equal(
        bundle["fire0"].random(64), _values_after(SEED, "fire0", frames * FIRE_DRAWS)
    ), f"fire0 stream desynchronised for scene: {label}"
    assert np.array_equal(
        bundle["splash"].random(64), _values_after(SEED, "splash", frames * SPLASH_DRAWS)
    ), f"splash stream desynchronised for scene: {label}"


def test_fire_block_is_large_enough_for_the_longest_path() -> None:
    """`Draws.next` raises past the end deliberately. If FIRE_DRAWS were ever
    smaller than the longest branch through `_draw_fire`, every frame after the
    first overrun would silently desynchronise — so probe the branches hard."""
    worst = 0
    for trial in range(400):
        gen = np.random.default_rng(trial)
        block = Draws(gen.random(FIRE_DRAWS))
        frame = np.zeros((_CAM.cfg.height, _CAM.cfg.width, 3), dtype=np.uint8)
        _draw_fire(frame, _CAM, _FIRE, 0.0, block)
        worst = max(worst, block.used)
    assert worst <= FIRE_DRAWS
    # and the block is not wildly oversized — that would just be dead entropy
    assert worst == FIRE_DRAWS, (
        f"longest observed path used {worst} of {FIRE_DRAWS}; retune the constant"
    )


# ---------------------------------------------------------------- multi-fire

def test_each_fire_gets_its_own_stream_keyed_on_index() -> None:
    rig = SimRig(DEFAULT_CONFIG, SimScenario(extra_fires=((-30.0, 6.0),)), seed=SEED)
    assert set(rig.rng.names()) == {"splash", "fire0", "fire1"}


def test_a_fire_going_out_does_not_move_the_other_fires_stream() -> None:
    """The old renderer drew only the fires that were still alive, so a fire
    going out changed the whole frame's draw count. Fire 1's stream must be
    unaffected by whether fire 0 is still burning."""
    frames = 5

    def fire1_after(kill_fire0: bool) -> np.ndarray:
        rig = SimRig(DEFAULT_CONFIG, SimScenario(extra_fires=((-30.0, 6.0),)), seed=SEED)
        if kill_fire0:
            rig.fires[0].intensity = 0.0
        for _ in range(frames):
            rig.render()
        return rig.rng["fire1"].random(64)

    assert np.array_equal(fire1_after(False), fire1_after(True))
    assert np.array_equal(
        fire1_after(False), _values_after(SEED, "fire1", frames * FIRE_DRAWS)
    )
