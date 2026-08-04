# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Plugin discovery — **provisional, v0**.

Third parties can supply rigs, mission profiles and detectors through Python
entry points. The interface is explicitly **not stable**: the group names carry
`v0` and will change. Freezing an interface around one implementation guarantees
it is wrong, which is why the dogfood rule below exists.

## Discovery does not import

`discover()` reads `importlib.metadata` **metadata only**. It never imports a
plugin module, so listing what is installed cannot execute third-party code. That
matters more than it sounds: plugin authors really do write `sys.exit()` and
network calls at module scope, and `triton --list-plugins` should not be able to
kill the process or phone home.

`load()` is the single place `ep.load()` runs, and it catches `BaseException` —
not `Exception`. `SystemExit` and `KeyboardInterrupt` are not `Exception`
subclasses, and a module-scope `sys.exit()` in somebody's plugin must not take
the turret's control process with it.

## Built-in names are unshadowable

`pip install evil-plugin` registering the name `fire` cannot replace the
flagship profile. This is the same rule as `configschema`'s built-in groups and
for the same reason: installing a package is not an acceptable way to acquire the
power to redefine a safety-relevant component.

## The dogfood rule

`SimRig`, `SerialRig`, `NullRig`, `FireProfile`, `WashdownProfile` and both
detectors are registered through this API and must load through it before v0 is
declared. If the built-ins cannot use the extension point, it is not an extension
point — it is a hole with documentation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

GROUP_RIG = "triton.rig.v0"
GROUP_PROFILE = "triton.profile.v0"
GROUP_DETECTOR = "triton.detector.v0"

GROUPS = (GROUP_RIG, GROUP_PROFILE, GROUP_DETECTOR)


@dataclass(frozen=True)
class PluginInfo:
    """What discovery can say WITHOUT importing anything."""

    group: str
    name: str
    value: str          # "package.module:attribute", straight from the metadata
    distribution: str = ""
    builtin: bool = False


class PluginError(RuntimeError):
    """A plugin failed to load. Never raised for a built-in."""


# name -> factory, per group. Populated at import by `_register_builtins`.
_BUILTINS: dict[str, dict[str, Callable[[], object]]] = {g: {} for g in GROUPS}


def register_builtin(group: str, name: str, factory: Callable[[], object]) -> None:
    if group not in GROUPS:
        raise ValueError(f"unknown plugin group {group!r}; known: {list(GROUPS)}")
    _BUILTINS[group][name] = factory


def builtin_names(group: str) -> tuple[str, ...]:
    return tuple(sorted(_BUILTINS.get(group, {})))


def discover(group: str) -> list[PluginInfo]:
    """Everything registered in `group`, built-ins first. **Imports nothing.**"""
    if group not in GROUPS:
        raise ValueError(f"unknown plugin group {group!r}; known: {list(GROUPS)}")

    found = [
        PluginInfo(group=group, name=name, value="<built-in>", builtin=True)
        for name in builtin_names(group)
    ]

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
        return found

    reserved = set(builtin_names(group))
    for ep in entry_points(group=group):
        if ep.name in reserved:
            # Unshadowable: a third party cannot take a built-in's name.
            continue
        dist = getattr(getattr(ep, "dist", None), "name", "") or ""
        found.append(PluginInfo(group=group, name=ep.name, value=ep.value, distribution=dist))
    return found


def load(group: str, name: str):
    """Instantiate a plugin. The ONLY place third-party code is imported.

    Built-ins resolve without touching `importlib.metadata` at all, so the
    flagship paths cannot be broken by a malformed third-party entry point.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown plugin group {group!r}; known: {list(GROUPS)}")

    factory = _BUILTINS.get(group, {}).get(name)
    if factory is not None:
        return factory()

    from importlib.metadata import entry_points

    for ep in entry_points(group=group):
        if ep.name != name:
            continue
        try:
            target = ep.load()
        except BaseException as exc:  # noqa: BLE001 - see the module docstring
            # BaseException, deliberately: a module-scope sys.exit() in a plugin
            # raises SystemExit, which is NOT an Exception, and must not take the
            # turret's process down with it.
            raise PluginError(
                f"plugin {name!r} in group {group!r} failed to load: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        try:
            return target()
        except BaseException as exc:  # noqa: BLE001
            raise PluginError(
                f"plugin {name!r} in group {group!r} failed to construct: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

    raise PluginError(f"no plugin named {name!r} in group {group!r}")


def _register_builtins() -> None:
    """The dogfood rule: every shipped implementation goes through the API."""
    from .profiles import FireProfile, WashdownProfile

    register_builtin(GROUP_PROFILE, "fire", FireProfile)
    register_builtin(GROUP_PROFILE, "washdown", WashdownProfile)

    def _null_rig():
        from .rig.interface import NullRig

        return NullRig()

    def _sim_rig():
        from .config import DEFAULT_CONFIG
        from .rig.sim_rig import SimRig, SimScenario

        return SimRig(DEFAULT_CONFIG, SimScenario())

    def _serial_rig_factory():
        # Returned uninstantiated: constructing it would open a serial port, and
        # discovery/loading must never actuate anything.
        from .rig.serial_rig import SerialRig

        return SerialRig

    register_builtin(GROUP_RIG, "null", _null_rig)
    register_builtin(GROUP_RIG, "sim", _sim_rig)
    register_builtin(GROUP_RIG, "serial", _serial_rig_factory)

    def _classical_detector():
        from .config import DEFAULT_CONFIG
        from .vision.firedetect import FireDetector

        return FireDetector(DEFAULT_CONFIG.detector)

    def _onnx_detector_factory():
        from .vision.onnx_detector import OnnxFireDetector

        return OnnxFireDetector

    register_builtin(GROUP_DETECTOR, "classical", _classical_detector)
    register_builtin(GROUP_DETECTOR, "onnx", _onnx_detector_factory)


_register_builtins()
