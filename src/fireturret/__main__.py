# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""FireTurret CLI.

  fireturret sim [--scenario NAME] [--seed N] [--headless] [--frames N] [--record out.mp4]
      Full end-to-end run against the built-in physics simulator.
      Scenarios: default, close, offset, far, unreachable, multi, windy, decoy, moving.

  fireturret run --source 0|video.mp4 [--port COM3] [--config turret.json] [--arm-water]
      Real footage. With --port, drives the hardware turret over serial (water
      stays OFF unless --arm-water is given and you confirm). Without --port, a
      NullRig gives a dry-run aim preview. --config loads a calibrated FireTurretConfig.

  fireturret selftest --port COM3 [--arm-water]
      Exercise each actuator and verify the telemetry link before a live mission.
      Water stays off unless --arm-water is given and you type ARM — the same
      gate as `run` and `web`.

  fireturret web [--source 0|video.mp4|sim] [--port COM3] [--http-port 8000] [--expose]
      Serve a browser interface (live view + warnings + E-stop). Binds to
      LOOPBACK by default: the console is unauthenticated and its E-stop
      latches SAFE with no rearm from the browser, so reaching it from another
      machine needs --host <addr> --expose. --source sim needs no hardware.

  fireturret fit shots.csv
      Calibrate velocity_coeff and drag_k from measured test shots
      (CSV rows: pump_pct,elevation_deg,measured_range_m).

  fireturret studio
      Launch FireTurret Studio, the desktop engineering workbench
      (needs the [desktop] extra: pip install fireturret[desktop]).
"""

from __future__ import annotations

import argparse
import csv
import sys

from .app import run_capture, run_sim
from .config import DEFAULT_CONFIG, load_config
from .rig.interface import NullRig
from .rig.sim_rig import SimScenario

SCENARIOS: dict[str, SimScenario] = {
    "default": SimScenario(),
    "close": SimScenario(fire_azimuth_deg=15.0, fire_range_m=5.0),
    "offset": SimScenario(fire_azimuth_deg=-35.0, fire_range_m=6.0, azimuth_bias_deg=-1.0,
                          velocity_coeff_scale=1.05, drag_scale=0.85),
    "far": SimScenario(fire_azimuth_deg=0.0, fire_range_m=8.5),
    "unreachable": SimScenario(fire_azimuth_deg=0.0, fire_range_m=10.0,
                               velocity_coeff_scale=0.88, drag_scale=1.3),
    "multi": SimScenario(fire_azimuth_deg=25.0, fire_range_m=7.0, extra_fires=((-30.0, 6.0),)),
    "windy": SimScenario(fire_azimuth_deg=20.0, fire_range_m=6.5, wind_cross_amp_deg=5.0,
                         wind_range_amp_m=1.5, wind_period_s=8.0),
    "decoy": SimScenario(fire_azimuth_deg=25.0, fire_range_m=6.5, decoy=(-30.0, 6.0)),
    "moving": SimScenario(fire_azimuth_deg=5.0, fire_range_m=5.0, fire_drift_az_dps=1.5,
                          fire_drift_range_mps=0.05),
}


def _cmd_sim(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    report = run_sim(
        cfg,
        scenario=SCENARIOS[args.scenario],
        seed=args.seed,
        headless=args.headless,
        max_frames=args.frames,
        record_path=args.record,
        record_telemetry=bool(args.log or args.report),
    )
    acq = report.time_to_first_suppress_s
    acq_str = f"{acq:.1f}s" if acq is not None else "-"
    print(
        f"extinguished={report.extinguished} sim_s={report.sim_seconds:.1f} "
        f"water_l={report.water_used_l:.2f} acquire={acq_str}"
    )
    print(
        f"  aim: mean_miss={report.mean_miss_m:.2f}m peak_miss={report.peak_miss_m:.2f}m "
        f"states={sorted(report.states_visited)}"
    )
    if report.advisories_raised:
        print(f"  operator advisories: {sorted(report.advisories_raised)}")
    if args.log:
        from .analysis import write_csv
        write_csv(report.telemetry, args.log)
        print(f"  telemetry -> {args.log} ({len(report.telemetry)} samples)")
    if args.report:
        from . import ballistics
        from .analysis import write_report
        lo, hi = ballistics.reach_bounds(
            cfg.jet, cfg.turret.tilt_min_deg, cfg.turret.tilt_max_deg,
        )
        write_report(report.telemetry, args.report, lo, hi,
                     title=f"FireTurret mission - {args.scenario}")
        print(f"  analysis report -> {args.report}")
    return 0 if report.extinguished else 1


def _confirm_arm() -> bool:
    """Interactive arm gate: only 'ARM' (typed exactly) enables water."""
    try:
        answer = input("Arm water? Type ARM to enable pump/valve (anything else = dry): ")
    except EOFError:
        return False
    return answer.strip() == "ARM"


def _open_serial_rig(port: str, cfg=None):
    """The ONE place any CLI path constructs a hardware rig — and the place the
    water gate is applied.

    Every command that can actuate a real turret comes through here, so the
    connection check and the error message cannot drift apart between `run`,
    `web` and `selftest`. The rig comes back wrapped in a `GuardedRig`, DISARMED,
    so a path that never constructs a `Pipeline` — the Studio, a manual jog, a
    future plugin — still cannot spray. Returns None when the turret could not be
    reached.
    """
    from .control.gate import standard_gate
    from .rig.guarded import GuardedRig
    from .rig.serial_rig import SerialRig

    try:
        rig = SerialRig(port)
    except ConnectionError as exc:
        print(f"error: {exc}")
        return None
    print(f"connected to turret on {port}")
    return GuardedRig(rig, standard_gate(cfg or DEFAULT_CONFIG), armed=False)


def _arm_water(args: argparse.Namespace) -> bool:
    """The ONE water-arming gate for every hardware path.

    Water is OFF unless the operator passed --arm-water *and* typed ARM.
    `selftest` goes through this too: its pump pulse is every bit as wet as a
    mission, and it used to need only a bare --water flag.
    """
    if getattr(args, "arm_water", False) and _confirm_arm():
        print("WATER ARMED — pump/valve live.")
        return True
    print(
        "dry-aim: pan/tilt move but pump/valve are DISABLED. "
        "Re-run with --arm-water (and type ARM) to enable water."
    )
    return False


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    source: int | str = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    water_enabled = False
    if args.port:
        rig = _open_serial_rig(args.port, cfg)
        if rig is None:
            return 2
        # SAFETY: on real hardware water stays OFF (dry-aim) unless explicitly armed.
        water_enabled = _arm_water(args)
        if water_enabled:
            rig.arm()  # the guard is disarmed by default; this is the only opener
    else:
        rig = NullRig()
        print("no --port given: dry-run aim preview (no actuation)")

    detector = None
    if args.detector == "onnx":
        if not args.model:
            print("--detector onnx requires --model PATH")
            return 2
        from .vision.onnx_detector import OnnxFireDetector

        try:
            detector = OnnxFireDetector(args.model, cfg.detector)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"error loading model: {exc}")
            return 2
        print(f"using learned ONNX detector: {args.model}")

    run_capture(
        cfg,
        source,
        rig,
        headless=args.headless,
        record_path=args.record,
        max_frames=args.frames,
        water_enabled=water_enabled,
        detector=detector,
    )
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    from .ui.webserver import serve

    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    rig = None
    scenario = None
    water_enabled = False
    if args.source == "sim":
        scenario = SCENARIOS.get(args.scenario, SCENARIOS["default"])
    elif args.port:
        rig = _open_serial_rig(args.port, cfg)
        if rig is None:
            return 2
        # SAFETY: on real hardware water stays OFF (dry-aim) unless explicitly armed.
        water_enabled = _arm_water(args)
        if water_enabled:
            rig.arm()
    try:
        serve(cfg, args.source, host=args.host, port=args.http_port,
              rig=rig, scenario=scenario, water_enabled=water_enabled,
              acknowledge_exposure=args.expose)
    except ValueError as exc:  # non-loopback bind without --expose
        print(f"error: {exc}")
        return 2
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    """Grid-search least-squares fit of velocity_coeff and drag_k against
    measured shots. Prints the best pair as a config snippet."""
    from . import ballistics
    from .config import DEFAULT_CONFIG

    shots: list[tuple[float, float, float]] = []
    with open(args.csv, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#") or not row[0].strip():
                continue
            if row[0].strip().lower() in ("pump_pct", "pump"):
                continue
            shots.append((float(row[0]), float(row[1]), float(row[2])))
    if len(shots) < 3:
        print("need at least 3 shots (pump_pct,elevation_deg,measured_range_m)")
        return 2

    try:
        fit = ballistics.fit_jet(shots, DEFAULT_CONFIG.jet)
    except ValueError as exc:
        # An unfittable set is bad input, not a crash: report it the same way the
        # shot-count guard above does.
        print(exc)
        return 2
    print(f"fit over {len(shots)} shots: rmse = {fit.rmse_m:.2f} m")
    print("update src/fireturret/config.py JetConfig with:")
    print(f"    velocity_coeff: float = {fit.velocity_coeff:.2f}")
    print(f"    drag_k: float = {fit.drag_k:.3f}")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Exercise each actuator and verify the telemetry link before a live mission."""
    from .selftest import run_selftest

    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    rig = _open_serial_rig(args.port, cfg)
    if rig is None:
        return 2
    # SAFETY: the self-test's pump pulse goes through the SAME arming gate as a
    # mission. It used to fire on a bare --water flag with no typed confirmation.
    water = _arm_water(args)
    if water:
        rig.arm()
    print(f"self-test on {args.port} (water {'ARMED' if water else 'OFF'})...")
    try:
        report = run_selftest(rig, cfg, water=water)
    finally:
        rig.close()
    for name, ok, detail in report.checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} {detail}")
    print("\nMANUAL FAILSAFE CHECKS before any live run:")
    print("  1. Press E-stop -> telemetry 'estop' must read 1 and pump/valve must cut.")
    print("  2. Kill the host -> with no heartbeat >0.5 s the board must cut pump/valve.")
    print("  3. Jog to each pan/tilt limit -> the geofence must keep the nozzle off people.")
    print(f"\nself-test {'PASSED' if report.passed else 'FAILED'}")
    return 0 if report.passed else 1


def _cmd_studio(args: argparse.Namespace) -> int:
    """Launch the desktop engineering workbench. Imported lazily so the core CLI
    never depends on PySide6 (needs the `fireturret[desktop]` extra)."""
    from .studio.app import main as studio_main

    return studio_main()


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from dispatch so tests can inspect defaults —
    notably the safety-relevant ones (`web --host`, `selftest --arm-water`)
    without connecting to a turret."""
    parser = argparse.ArgumentParser(prog="fireturret", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("sim", help="run against the built-in simulator")
    p_sim.add_argument("--scenario", choices=sorted(SCENARIOS), default="default",
                       help="which built-in scenario to run")
    p_sim.add_argument("--seed", type=int, default=7)
    p_sim.add_argument("--headless", action="store_true")
    p_sim.add_argument("--frames", type=int, default=6000)
    p_sim.add_argument("--record", default=None, help="write an annotated .mp4")
    p_sim.add_argument("--log", default=None, help="export per-frame telemetry to CSV")
    p_sim.add_argument("--report", default=None, help="render an analysis figure to PNG (needs matplotlib)")
    p_sim.add_argument("--config", default=None, help="load a calibrated FireTurretConfig JSON (Studio export or hand-edited)")
    p_sim.set_defaults(func=_cmd_sim)

    p_run = sub.add_parser("run", help="run on a webcam or video file")
    p_run.add_argument("--source", required=True, help="webcam index or video path")
    p_run.add_argument("--port", default=None, help="serial port of the turret (e.g. COM3, /dev/ttyUSB0)")
    p_run.add_argument("--headless", action="store_true")
    p_run.add_argument("--frames", type=int, default=None)
    p_run.add_argument("--record", default=None)
    p_run.add_argument("--config", default=None, help="load a calibrated FireTurretConfig JSON")
    p_run.add_argument("--arm-water", action="store_true",
                       help="enable pump/valve on hardware (asks for confirmation; default is dry-aim)")
    p_run.add_argument("--detector", choices=["classical", "onnx"], default="classical",
                       help="fire detector: classical (default) or a learned ONNX model")
    p_run.add_argument("--model", default=None, help="ONNX model path (with --detector onnx)")
    p_run.set_defaults(func=_cmd_run)

    p_web = sub.add_parser("web", help="serve a browser interface (live view + E-stop)")
    p_web.add_argument("--source", default="sim", help="'sim', a webcam index, or a video path")
    p_web.add_argument("--scenario", choices=sorted(SCENARIOS), default="default",
                       help="scenario when --source sim")
    p_web.add_argument("--port", default=None, help="serial port of the turret (real hardware)")
    p_web.add_argument("--arm-water", action="store_true",
                       help="enable pump/valve on hardware (asks for confirmation; default is dry-aim)")
    p_web.add_argument("--host", default="127.0.0.1",
                       help="bind address (loopback by default; see --expose)")
    p_web.add_argument("--expose", action="store_true",
                       help="acknowledge that binding a non-loopback address exposes an "
                            "unauthenticated camera stream and a remote E-stop")
    p_web.add_argument("--http-port", type=int, default=8000)
    p_web.add_argument("--config", default=None, help="load a calibrated FireTurretConfig JSON")
    p_web.set_defaults(func=_cmd_web)

    p_fit = sub.add_parser("fit", help="calibrate jet parameters from test shots")
    p_fit.add_argument("csv", help="CSV of pump_pct,elevation_deg,measured_range_m")
    p_fit.set_defaults(func=_cmd_fit)

    p_selftest = sub.add_parser(
        "selftest", help="exercise the turret actuators and verify the link (needs --port)")
    p_selftest.add_argument("--port", required=True, help="serial port of the turret (e.g. COM3)")
    p_selftest.add_argument("--arm-water", action="store_true",
                            help="also pulse pump/valve briefly (asks for confirmation; "
                                 "default is dry)")
    p_selftest.add_argument("--config", default=None, help="load a calibrated FireTurretConfig JSON")
    p_selftest.set_defaults(func=_cmd_selftest)

    p_studio = sub.add_parser(
        "studio", help="launch the desktop engineering workbench (needs fireturret[desktop])"
    )
    p_studio.set_defaults(func=_cmd_studio)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
