# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The config schema registry.

`config.py` names its six sub-configs in one dict literal. That is fine while
there are six and one author. It stops being fine the moment four workstreams
each need to add one — every branch edits the same line, and a plugin author
cannot add one at all without patching the package.

So groups are *registered* rather than listed. Built-ins register themselves at
import; anything else registers through `register_subconfig`.

## What a registered group buys over the raw passthrough

SP1 already made `config_from_dict` preserve groups it does not recognise, so a
plugin's settings survive a load/save round trip. That is the floor, not the
goal: an unrecognised group is carried as an opaque dict, so nothing validates
it, a typo in it is silent, and it round-trips a `0.5` that should have been a
`(0.5, 1.0)` without complaint.

Registering the group's dataclass instead means it is rebuilt through its own
constructor on load — so `__post_init__` runs and a bad value fails loudly, in
the same place and the same way a bad built-in value does.

## Built-ins are unshadowable

`register_subconfig` refuses to replace `turret`, `jet` and friends. A plugin
that could redefine the geofence sub-config could disable the geofence, and
"install this package" is not an acceptable way to acquire that power.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

# The groups that map onto FireTurretConfig's own fields. Populated by config.py at
# import time; a name in here can never be re-registered.
BUILTIN_GROUPS: dict[str, type] = {}

_PLUGIN_GROUPS: dict[str, type] = {}

# Reserved top-level keys that are not sub-config groups.
RESERVED_KEYS = frozenset({"schema_version", "config"})


def register_builtin(name: str, cls: type) -> None:
    """Register one of the package's own sub-configs. Called by `config.py`."""
    BUILTIN_GROUPS[name] = cls


def register_subconfig(name: str, cls: type) -> None:
    """Register a third-party sub-config group.

    Raises rather than overwriting, in both directions that matter: a plugin may
    not shadow a built-in (that would let an install disable a safety geofence),
    and two plugins may not claim the same group name with different classes
    (that would make load order decide which validation runs).
    """
    if name in RESERVED_KEYS:
        raise ValueError(f"{name!r} is a reserved config key")
    if name in BUILTIN_GROUPS:
        raise ValueError(
            f"{name!r} is a built-in config group and cannot be replaced. "
            "Built-ins are unshadowable so that installing a package can never "
            "redefine a safety-relevant sub-config."
        )
    existing = _PLUGIN_GROUPS.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"config group {name!r} is already registered to {existing.__name__}"
        )
    if not is_dataclass(cls):
        raise TypeError(
            f"config group {name!r} must be a dataclass so it can be validated "
            "and serialised like every built-in group"
        )
    _PLUGIN_GROUPS[name] = cls


def unregister_subconfig(name: str) -> None:
    """Drop a plugin group. Exists for tests and for a plugin being unloaded;
    built-ins cannot be removed."""
    _PLUGIN_GROUPS.pop(name, None)


def plugin_groups() -> dict[str, type]:
    return dict(_PLUGIN_GROUPS)


def all_groups() -> dict[str, type]:
    """Built-ins first, so a serialised config keeps a stable, readable order
    regardless of what happens to be installed."""
    return {**BUILTIN_GROUPS, **_PLUGIN_GROUPS}


def coerce_group(name: str, raw: dict) -> object:
    """Rebuild a registered group from its serialised form.

    Unknown fields are dropped rather than fatal — the same tolerance built-in
    groups get, and for the same reason: a config file is edited by humans and
    written by tools of different vintages. Values that ARE present go through
    the dataclass constructor, so `__post_init__` still rejects nonsense.
    """
    cls = _PLUGIN_GROUPS.get(name)
    if cls is None:
        raise KeyError(f"config group {name!r} is not registered")
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in allowed})
