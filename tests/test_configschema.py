# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP3 — the config schema registry.

Two things are being bought here. The first is boring and mechanical: four
workstreams can each add a sub-config without all four editing the same dict
literal. The second is the one that matters — a REGISTERED group is validated on
load, where an unregistered one is only preserved.

And one negative property: built-in groups are unshadowable. A plugin that could
re-register `turret` could redefine the keep-out geofence, and `pip install` is
not an acceptable way to acquire that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from triton import configschema
from triton.config import (
    DEFAULT_CONFIG,
    config_from_dict,
    config_to_dict,
    plugin_config,
)


@dataclass(frozen=True)
class DemoPluginConfig:
    threshold: float = 0.25
    label: str = "demo"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Registration is process-global, so every test unwinds its own."""
    yield
    for name in list(configschema.plugin_groups()):
        configschema.unregister_subconfig(name)


# ------------------------------------------------------------------ registry

def test_a_registered_group_is_validated_on_load() -> None:
    """The whole point. An unregistered group round-trips a nonsense value in
    silence; a registered one fails in the same place a built-in would."""
    configschema.register_subconfig("demo", DemoPluginConfig)
    d = config_to_dict(DEFAULT_CONFIG)
    d["demo"] = {"threshold": 5.0}  # outside [0, 1]
    with pytest.raises(ValueError, match="threshold"):
        config_from_dict(d)


def test_a_registered_group_round_trips_as_its_own_type() -> None:
    configschema.register_subconfig("demo", DemoPluginConfig)
    d = config_to_dict(DEFAULT_CONFIG)
    d["demo"] = {"threshold": 0.5, "label": "calibrated"}

    cfg = config_from_dict(d)
    loaded = plugin_config(cfg, "demo")
    assert isinstance(loaded, DemoPluginConfig)
    assert loaded.threshold == 0.5
    assert loaded.label == "calibrated"
    assert config_to_dict(cfg)["demo"] == {"threshold": 0.5, "label": "calibrated"}


def test_an_unknown_field_in_a_registered_group_is_dropped_not_fatal() -> None:
    """Same tolerance the built-ins get: a config file is edited by humans and
    written by tools of different vintages."""
    configschema.register_subconfig("demo", DemoPluginConfig)
    d = config_to_dict(DEFAULT_CONFIG)
    d["demo"] = {"threshold": 0.5, "field_from_a_later_version": 1}
    assert plugin_config(config_from_dict(d), "demo").threshold == 0.5


def test_an_unregistered_group_is_still_preserved_verbatim() -> None:
    """SP1's floor still holds: settings from a plugin that is not currently
    installed must survive a load/save cycle rather than being destroyed."""
    d = config_to_dict(DEFAULT_CONFIG)
    d["not_installed"] = {"anything": [1, 2, 3]}
    assert config_to_dict(config_from_dict(d))["not_installed"] == {"anything": [1, 2, 3]}


# --------------------------------------------------------------- unshadowable

@pytest.mark.parametrize("builtin", ["camera", "turret", "jet", "detector", "servo", "mission"])
def test_builtin_groups_cannot_be_shadowed(builtin) -> None:
    """A plugin that could redefine `turret` could disable the geofence."""
    with pytest.raises(ValueError, match="built-in"):
        configschema.register_subconfig(builtin, DemoPluginConfig)


def test_reserved_keys_cannot_be_registered() -> None:
    with pytest.raises(ValueError, match="reserved"):
        configschema.register_subconfig("schema_version", DemoPluginConfig)


def test_two_plugins_cannot_claim_one_name() -> None:
    """Otherwise import order decides whose validation runs."""
    @dataclass(frozen=True)
    class Other:
        x: int = 0

    configschema.register_subconfig("demo", DemoPluginConfig)
    configschema.register_subconfig("demo", DemoPluginConfig)  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        configschema.register_subconfig("demo", Other)


def test_a_non_dataclass_group_is_rejected() -> None:
    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="dataclass"):
        configschema.register_subconfig("demo", NotADataclass)


# ------------------------------------------------------------- built-ins hold

def test_registering_a_plugin_does_not_disturb_the_builtin_round_trip() -> None:
    configschema.register_subconfig("demo", DemoPluginConfig)
    assert config_from_dict(config_to_dict(DEFAULT_CONFIG)) == DEFAULT_CONFIG


def test_builtins_are_all_registered() -> None:
    assert set(configschema.BUILTIN_GROUPS) == {
        "camera", "turret", "jet", "detector", "servo", "mission",
    }
