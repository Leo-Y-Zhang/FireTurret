# Commissioning & safe bring-up

A gated procedure for taking a **built** turret from "parts assembled" to a
first supervised live test, with a safety gate at every step. Read this together
with [`BUILD_GUIDE.md`](BUILD_GUIDE.md) (mechanical build + BOM) and the safety
notes in the [`README`](../README.md).

> **This is a supervised demonstrator, not a life-safety device.** Water is wrong
> for electrical and grease fires. Keep a real extinguisher within reach, never
> leave it running unattended, and geofence the aim so the nozzle can never point
> at a person. Do every step in order; do not skip a gate.

## What the software already enforces

- **Control-layer arming** — the mission controller itself refuses to command
  pump/valve unless it is **armed**. The simulator/CLI arm it; a real hardware run
  stays disarmed (dry-aim) until you explicitly enable water. This is the safety
  authority, not just a CLI flag.
- **Fault latching** — an E-stop or telemetry loss latches the mission into SAFE
  and **disarms water**; it does **not** auto-resume when the fault clears. The
  operator must deliberately re-arm. (A passing blip can no longer restart autonomy
  and spray on its own.)
- **Connection verification** — `SerialRig` refuses to run unless the board is
  actually sending telemetry (a port that opens but is the wrong device/baud, or
  has no firmware, raises a clear error instead of looking "connected").
- **Dry-aim by default** — `triton run --port …` tracks the fire with pan/tilt but
  keeps pump/valve OFF (both in the control layer and as a per-command backstop).
  Water is only enabled with `--arm-water`, after you type `ARM` at the prompt.
- **Homing gate (firmware)** — the pan angle is dead-reckoned, so the firmware
  keeps water locked until the pan axis is **homed** against its limit switch; an
  un-homed turret cannot spray.
- **Self-test** — `triton selftest --port …` exercises each actuator and verifies
  the link before any autonomous run. Its pump pulse goes through the **same**
  arming gate as a mission (`--arm-water` plus a typed `ARM`), and it will not aim
  into a configured keep-out sector — it drives the rig directly, so it does not
  inherit `MissionController`'s geofence and must enforce its own.
- **Firmware failsafes** — an independent hardware E-stop and a 500 ms heartbeat
  watchdog cut pump/valve regardless of the host.
- **Calibration as data** — `--config turret.json` loads your calibration from a
  file (export it from Studio) instead of editing source.

## The procedure

Each step lists the command, what "pass" looks like, and the **gate** you must
clear before moving on.

### 1. Bench test — no water, motors unpowered
Flash `firmware/turret_firmware/turret_firmware.ino`, set the pin map and
mechanical limits (`PAN_MIN/MAX`, `TILT_MIN/MAX`, `PAN_STEPS_PER_DEG`). Power the
logic only (not the pump/motor 12 V rail yet).
- **Pass:** `triton selftest --port COM3` reports `telemetry link: PASS`.
- **Gate:** if there's no telemetry, fix baud (115200) / wiring / firmware before anything moves.

### 2. Dry actuator test — motors powered, water line disconnected
Physically disconnect or empty the water line. Power the actuators.
- Run `triton selftest --port COM3` (water stays OFF).
- **Pass:** pan nudges ±5°, tilt sweeps, the warn indicator lights — and you
  **watch** each happen and confirm direction.
- **Gate:** if pan moves the wrong way or the wrong amount, fix `PAN_DIR_PIN` /
  `PAN_STEPS_PER_DEG` before continuing. See "Homing" below — the pan angle is
  dead-reckoned from power-on, so start from a known centre.

### 3. Verify the failsafes — by hand
These protect everything after, so prove them physically:
- **E-stop:** press it → telemetry `estop` must read 1 and pump/valve must be dead.
- **Heartbeat:** kill the host process → with no command for >0.5 s the board must
  cut pump/valve on its own.
- **Geofence:** jog to each pan/tilt limit and confirm the nozzle physically
  **cannot** point at a person or anything flammable.
- **Gate:** do not put water in the loop until all three pass.

### 4. Calibrate
Follow [`BUILD_GUIDE.md` §5](BUILD_GUIDE.md): camera field-of-view, mount
height/pitch, turret limits, and the jet model. Fire a dozen test shots at known
pump % / tilt, record where they land, and fit:
```bash
python -m triton fit shots.csv
```
Put the values in a config file (or use Studio's calibration panel and
export the config), then load it with `--config turret.json`.
- **Gate:** re-run `python -m triton sim --config turret.json` and confirm the
  simulated mission still behaves sensibly with your numbers.

### 5. Dry-aim over the real camera — still no water
```bash
python -m triton run --source 0 --config turret.json --port COM3
```
Water is OFF by default. Present a **safe test flame** (a candle at a safe distance).
- **Pass:** the turret detects the flame, tracks it, and the overlay shows a
  sensible aim solution — with pump/valve reading zero the whole time.
- **Gate:** only proceed if aim tracking looks correct and stable.

### 6. First armed live test — supervised
Outdoors or over a tray, extinguisher in hand, finger on the E-stop:
```bash
python -m triton run --source 0 --config turret.json --port COM3 --arm-water
```
Type `ARM` to enable water. Use a **small, controlled** flame.
- **Pass:** it drives the splash onto the flame and puts it out.
- If anything looks wrong, hit the E-stop first and ask questions later.

## Safety design notes

The guardrails above are additive to the firmware failsafes. Two are the
integrator's responsibility on a real build:

- **A keep-out sector, not just min/max.** The default pan window is nearly a full
  circle and tilt goes near-horizontal. For any real install, narrow
  `TurretConfig` pan/tilt to a window that physically cannot sweep across people,
  and verify it by jogging to the limits with water disabled.
- **First-power-on lockout.** For the very first motion tests, keep pump/valve
  physically disconnected (or jumpered off) so nothing can spray while you confirm
  the mechanics.

The homing gate, the fault-latching SAFE state (no auto-resume), and the arm gate
described in "What the software already enforces" are now implemented in the
firmware and control layer — you still wire a pan **home limit switch**
(`HOME_SWITCH_PIN`) for homing to function.
