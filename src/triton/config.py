# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Central configuration. Every physical constant that a real build must
calibrate lives here, with the calibration step named in docs/BUILD_GUIDE.md.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from . import configschema


@dataclass(frozen=True)
class CameraConfig:
    width: int = 960
    height: int = 540
    hfov_deg: float = 66.0  # calibrate: measured horizontal field of view
    mount_height_m: float = 0.55  # calibrate: lens centre above ground
    mount_pitch_deg: float = -8.0  # calibrate: fixed downward pitch on the pan stage
    # calibrate (boresight): azimuth offset between camera centre and nozzle
    boresight_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width/height must be positive")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError("hfov_deg must be in (0, 180)")
        if self.mount_height_m <= 0:
            raise ValueError("mount_height_m must be positive")


@dataclass(frozen=True)
class TurretConfig:
    pan_min_deg: float = -170.0
    pan_max_deg: float = 170.0
    tilt_min_deg: float = 8.0
    tilt_max_deg: float = 50.0
    pan_rate_dps: float = 90.0  # actuator slew rates (sim + sanity limits)
    tilt_rate_dps: float = 60.0
    pump_rate_pct_ps: float = 120.0  # pump ramp, percent per second
    # optional keep-out sector (pan_lo, pan_hi) the turret must NEVER aim into,
    # e.g. a doorway or where people stand. None = no keep-out.
    pan_keepout_deg: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        # a misconfigured turret is a safety issue (aim geofence); fail loudly
        if not self.pan_min_deg < self.pan_max_deg:
            raise ValueError("pan_min_deg must be < pan_max_deg")
        if not 0.0 <= self.tilt_min_deg < self.tilt_max_deg <= 90.0:
            raise ValueError("require 0 <= tilt_min < tilt_max <= 90")
        if min(self.pan_rate_dps, self.tilt_rate_dps, self.pump_rate_pct_ps) <= 0:
            raise ValueError("actuator rates must be positive")
        if self.pan_keepout_deg is not None:
            lo, hi = self.pan_keepout_deg
            if not self.pan_min_deg <= lo < hi <= self.pan_max_deg:
                raise ValueError("pan_keepout_deg must be an ascending sub-range of the pan limits")


@dataclass(frozen=True)
class JetConfig:
    """Pressure→velocity and drag model. All three are refined by `triton fit`."""

    max_pressure_psi: float = 60.0
    velocity_coeff: float = 0.90  # nozzle discharge/velocity coefficient
    drag_k: float = 0.16  # quadratic drag per metre (jet breakup), 1/m
    nozzle_height_m: float = 0.60  # nozzle exit above ground
    min_pump_pct: float = 20.0  # below this the stream is not coherent

    def __post_init__(self) -> None:
        if self.max_pressure_psi <= 0 or self.velocity_coeff <= 0:
            raise ValueError("max_pressure_psi and velocity_coeff must be positive")
        if not 0.0 < self.min_pump_pct < 100.0:
            raise ValueError("min_pump_pct must be in (0, 100)")
        if self.drag_k < 0 or self.nozzle_height_m < 0:
            raise ValueError("drag_k and nozzle_height_m must be non-negative")


@dataclass(frozen=True)
class DetectorConfig:
    min_blob_area_px: int = 60
    flicker_window: int = 12  # frames of mask history
    flicker_threshold: float = 0.08  # mean per-pixel toggle rate for "fire-like"
    confirm_hits: int = 8  # tracker persistence before a target is real
    max_misses: int = 10
    # HSV gates (OpenCV ranges: H 0-179, S/V 0-255)
    h_max: int = 25
    s_min: int = 90
    v_min: int = 140


@dataclass(frozen=True)
class ServoConfig:
    # ALIGN (camera-centring) azimuth control
    pan_gain: float = 0.55  # fraction of measured azimuth error applied per step
    pan_deadband_deg: float = 0.6
    # SUPPRESS closed-loop control (splash-relative)
    suppress_tilt_deg: float = 22.0  # fixed low-arc elevation; pump controls range
    suppress_pan_gain: float = 0.4  # gentle: corrects boresight from splash offset
    suppress_pan_deadband_deg: float = 0.7
    # Range is servoed in pure IMAGE space: drive the splash's pixel-row onto
    # the fire's ground pixel-row. No absolute-range estimate is trusted — that
    # is the essence of "don't measure distance".
    range_gain_pct_per_px: float = 0.06  # pump correction per pixel of splash-y error
    range_deadband_px: float = 12.0
    pump_step_pct: float = 3.5  # max pump change per control step
    settle_frames: int = 5


@dataclass(frozen=True)
class MissionConfig:
    search_pan_rate_dps: float = 25.0
    acquire_timeout_s: float = 4.0
    align_timeout_s: float = 6.0
    max_spray_s: float = 45.0  # continuous spray limit before re-evaluation
    soak_s: float = 2.0
    confirm_clear_s: float = 3.0
    heartbeat_timeout_s: float = 0.5
    default_range_m: float = 8.0  # bootstrap when ground-plane estimate unavailable
    unreachable_cycles: int = 4  # pump-saturated-and-short cycles ⇒ out of reach
    hold_retry_s: float = 6.0  # HOLD dwell before retrying an unreachable target


@dataclass(frozen=True)
class TritonConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    turret: TurretConfig = field(default_factory=TurretConfig)
    jet: JetConfig = field(default_factory=JetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)


DEFAULT_CONFIG = TritonConfig()

# --- calibration as data: load/save a TritonConfig to a JSON file --------------
# Files are interchangeable with Studio session files (which nest the config under
# a "config" key). This keeps a real build's calibration in a file, not in source.

_SUBCONFIGS: dict[str, type] = {
    "camera": CameraConfig,
    "turret": TurretConfig,
    "jet": JetConfig,
    "detector": DetectorConfig,
    "servo": ServoConfig,
    "mission": MissionConfig,
}

# Register the built-ins so `configschema` can refuse to let anything shadow
# them. A plugin able to redefine `turret` could redefine the geofence.
for _name, _cls in _SUBCONFIGS.items():
    configschema.register_builtin(_name, _cls)


# Bumped whenever a sub-config gains or loses a field. Readers warn (but still
# load) when handed a file from a NEWER schema, so a config written by a later
# version degrades to "the fields I understand" rather than crashing.
SCHEMA_VERSION = 1

# Groups this version does not know about are carried through a load/save cycle
# untouched. Third-party plugin config lives in such groups, and silently
# dropping them destroys a user's settings on every save.
_EXTRA_GROUPS_ATTR = "_extra_groups"


def config_to_dict(cfg: TritonConfig) -> dict:
    d: dict = {"schema_version": SCHEMA_VERSION}
    d.update({key: asdict(getattr(cfg, key)) for key in _SUBCONFIGS})
    # Registered plugin groups are typed dataclasses; unregistered ones are the
    # opaque dicts SP1 taught the loader to preserve. Both survive a round trip.
    for key, value in getattr(cfg, _EXTRA_GROUPS_ATTR, {}).items():
        d[key] = asdict(value) if is_dataclass(value) else value
    return d


def config_from_dict(d: dict) -> TritonConfig:
    """Rebuild a config, tolerating anything a future or third-party writer added.

    Deliberately forgiving in three directions, because a calibration file is
    edited by humans and written by plugins:
      - a MISSING group falls back to that sub-config's defaults,
      - an UNKNOWN field inside a known group is ignored,
      - an UNKNOWN group is preserved verbatim for the next save.
    Values that ARE present are still validated by each sub-config's
    __post_init__ — tolerance is about shape, never about physical sanity.
    """
    # accept either a bare config dict or a session dict with a nested "config"
    if "config" in d and isinstance(d["config"], dict):
        d = d["config"]

    version = d.get("schema_version", SCHEMA_VERSION)
    if isinstance(version, int) and version > SCHEMA_VERSION:
        warnings.warn(
            f"config schema_version {version} is newer than this build understands "
            f"({SCHEMA_VERSION}); unknown fields will be ignored",
            UserWarning,
            stacklevel=2,
        )

    groups: dict[str, dict] = {}
    for key, cls in _SUBCONFIGS.items():
        raw = d.get(key)
        if not isinstance(raw, dict):
            groups[key] = {}
            continue
        allowed = {f.name for f in fields(cls)}
        groups[key] = {k: v for k, v in raw.items() if k in allowed}

    # JSON turns the keep-out tuple into a list; coerce it back so == round-trips
    keepout = groups["turret"].get("pan_keepout_deg")
    if keepout is not None:
        groups["turret"]["pan_keepout_deg"] = tuple(keepout)

    # each sub-config is rebuilt through its constructor, so __post_init__ validates
    cfg = TritonConfig(**{key: cls(**groups[key]) for key, cls in _SUBCONFIGS.items()})

    known = set(_SUBCONFIGS) | {"schema_version"}
    extras: dict[str, object] = {}
    registered = configschema.plugin_groups()
    for key, value in d.items():
        if key in known:
            continue
        if key in registered and isinstance(value, dict):
            # A REGISTERED group is rebuilt through its own dataclass, so its
            # __post_init__ rejects nonsense exactly as a built-in's would.
            # Unregistered groups stay opaque: preserved, but unvalidated.
            extras[key] = configschema.coerce_group(key, value)
        else:
            extras[key] = value
    if extras:
        object.__setattr__(cfg, _EXTRA_GROUPS_ATTR, extras)
    return cfg


def plugin_config(cfg: TritonConfig, name: str):
    """The registered plugin sub-config `name`, or None if this config has none.

    Plugins cannot add FIELDS to `TritonConfig` — it is a fixed dataclass — so
    their groups live alongside it and are reached through here.
    """
    return getattr(cfg, _EXTRA_GROUPS_ATTR, {}).get(name)


def save_config(path, cfg: TritonConfig) -> None:
    Path(path).write_text(json.dumps(config_to_dict(cfg), indent=2))


def load_config(path) -> TritonConfig:
    return config_from_dict(json.loads(Path(path).read_text()))
