# FireTurret — build guide

Everything needed to turn the software into a working fire-suppression turret.
Read the **[Safety](#safety)** section first — this build mixes water,
electronics, and moving parts.

> FireTurret is a supervised demonstrator, **not a life-safety
> device.** It does not replace smoke alarms, extinguishers, or sprinklers. Water must never be used
> on electrical or grease fires. Only ever run it attended, on a small controlled
> test flame (e.g. a tealight in a metal tray), outdoors or over a drip tray.

---

## 1. Bill of materials

Prices are rough UK ballparks and move around. This is the **recommended
self-contained build** (portable, closed-loop, fixed turret).

| Subsystem | Part | ~£ |
|---|---|---|
| Brain | Raspberry Pi 5 (8 GB) + 27 W PSU + 64 GB microSD + active cooler | 110 |
| Real-time actuation | Arduino Uno R3 (or Nano / ESP32) — runs `turret_firmware` | 15 |
| Camera | Raspberry Pi Camera Module 3 (wide) **or** Logitech C920 USB webcam | 30 |
| Pan drive | NEMA-17 stepper + GT2 belt & pulleys (or a worm-gear reduction) | 15 |
| Tilt drive | DS3218 20 kg·cm metal-gear servo | 15 |
| Stepper driver | TMC2209 (quiet, microstepping) + a small carrier/CNC shield | 10 |
| Pan bearing | Lazy-susan / turntable bearing | 8 |
| Pump | 12 V diaphragm pump, ~60 PSI, 5 L/min | 25 |
| Flow control | 12 V solenoid valve + logic-level MOSFET (IRLZ44N) | 12 |
| Nozzle + plumbing | Adjustable jet nozzle + 8 mm hose + fittings | 15 |
| Reservoir | 10–20 L container | 12 |
| Power | 12 V 5 A supply **or** 12 V LiFePO4 battery + 12→5 V buck converter | 35 |
| **Safety** | **Normally-closed emergency-stop button** + inline fuse holder + 5 A fuse | 12 |
| Frame | 2020 aluminium extrusion + brackets + a waterproof box for electronics | 45 |
| Misc | Dupont wires, screw terminals, heat-shrink, cable glands | 15 |
| **Recommended total** | | **~£375** |

**Budget (~£180–220):** use a laptop/PC you already own as the brain + the
Arduino for motors, a USB webcam, two high-torque servos instead of a stepper
pan, a smaller pump, and a 3D-printed/plywood frame.

**Premium (~£700–900):** Jetson Orin Nano Super (GPU for a learned fire detector)
+ an OAK-D depth/AI camera + geared steppers with homing limit switches + a
higher-flow pump + LiFePO4 battery + a proper IP-rated enclosure.

**Optional upgrades:** a FLIR Lepton 3.5 thermal camera (~£200) makes detection
near-trivial and works through smoke; a co-boresighted laser diode gives the
closed loop something bright to servo on before water even flows.

---

## 2. Wiring

Keep the 12 V (motor/pump) and 5 V (logic) supplies **separate with a common
ground**. Fuse the 12 V line. Route the emergency-stop so it physically cuts 12 V
to the pump and motor driver — not just a signal to the software.

```
                       ┌─────────── 12 V supply ───────────┐
                       │            (fused, via E-STOP)     │
                       ▼                                    ▼
                 [MOSFET]──pump                       [TMC2209]──NEMA-17 (pan)
                 [MOSFET]──solenoid valve                  ▲
                       ▲                                    │ STEP/DIR
   Raspberry Pi 5      │ gate (5 V logic)                   │
   (runs the host)     │                                    │
     │  USB serial     ├───── Arduino/ESP32 ────────────────┤
     └─────────────────┤   (runs turret_firmware)           │
                        ├──── PWM ──► tilt servo (DS3218)    │
                        ├──── GPIO ─► laser (optional)       │
                        └──── E-STOP sense (INPUT_PULLUP) ◄──┘  (NC to GND)
   Camera ── CSI/USB ── Raspberry Pi
```

Firmware pin map (edit at the top of `firmware/turret_firmware/turret_firmware.ino`):

| Signal | Default pin |
|---|---|
| Pan STEP / DIR | D2 / D5 |
| Tilt servo (PWM) | D9 |
| Pump PWM (MOSFET gate) | D3 |
| Valve (MOSFET/relay) | D7 |
| Laser (optional) | D8 |
| E-stop sense (active-low) | D4 (`INPUT_PULLUP`, NC switch to GND) |

---

## 3. Assembly

1. Build the base and mount the lazy-susan bearing; couple the NEMA-17 to the
   upper plate with the GT2 belt. A reduction (belt or worm) improves aiming
   resolution and holding torque against the hose's reaction force.
2. Mount the tilt servo on the rotating plate; attach the nozzle to the tilt arm.
3. **Mount the camera on the pan plate**, rigidly, looking along the nozzle with
   a small fixed downward pitch (~8°). The whole design depends on the camera and
   nozzle sharing the pan axis — the closed loop learns and cancels the small
   fixed boresight offset between them, but they must rotate together.
4. Plumb reservoir → pump → solenoid valve → nozzle. Keep all electronics in a
   sealed box with cable glands and drip loops; keep them away from the spray.
5. Wire per the diagram. Double-check the E-stop cuts 12 V before first power-on.

---

## 4. Firmware

1. Open `firmware/turret_firmware/turret_firmware.ino` in the Arduino IDE.
2. Set the pin map and the mechanical limits (`PAN_MIN/MAX`, `TILT_MIN/MAX`,
   `PAN_STEPS_PER_DEG`) to match your build.
3. Flash the board. It speaks the protocol in `src/triton/rig/protocol.py` at
   115200 baud, reports telemetry at 10 Hz, and independently enforces the
   E-stop and a 500 ms heartbeat watchdog (pump/valve cut if commands stop).

---

## 5. Calibration

Values live in `src/triton/config.py`. Calibrate in this order:

1. **Camera FOV** (`CameraConfig.hfov_deg`): point at two marks a known distance
   apart at a known range; compute the horizontal field of view. Set `width`/
   `height` to your capture resolution.
2. **Mount height & pitch** (`mount_height_m`, `mount_pitch_deg`): measure the
   lens centre height and its fixed downward angle. These set the opening-shot
   range estimate (the closed loop tolerates error here).
3. **Turret limits** (`TurretConfig`): the real mechanical pan/tilt range and the
   slew rates your motors achieve. Geofence pan/tilt so the turret can **never**
   point at a person.
4. **Jet model** (`JetConfig`): fire a dozen test shots at known pump % and tilt,
   measure where each lands, put them in a CSV (`pump_pct,elevation_deg,measured_range_m`),
   and run `python -m triton fit shots.csv`. Paste the printed `velocity_coeff`
   and `drag_k` back into `JetConfig`. Also set `max_pressure_psi`, `nozzle_height_m`,
   and `min_pump_pct` (the lowest pump that still throws a coherent stream).
5. **Boresight** (`boresight_offset_deg`): optional. The suppression loop learns
   and cancels the camera-to-nozzle offset from the splash, so a rough value is
   fine; set it only to speed up the first shot.

Verify end to end in the simulator first (`python -m triton sim`), then dry-run on
your webcam (`python -m triton run --source 0`, no `--port`) to confirm detection
and the aim solution before connecting hardware.

---

## 6. Safety

- **Emergency stop is mandatory** and must physically cut 12 V to the pump and
  motors (normally-closed, so a broken wire fails safe). The firmware also cuts
  pump/valve on E-stop and on loss of the host heartbeat.
- **Low-voltage only** for actuation (12 V DC). Keep mains well away from water;
  use an RCD if any mains is nearby. Enclose and waterproof all electronics; add
  drip loops and cable glands.
- **Geofence the aim** in `TurretConfig` so the turret cannot point at people, and
  never leave it running unattended.
- **Fuse the battery**; don't over-pressurise the plumbing.
- **Right agent for the fire:** water only, and never on electrical or grease
  fires. This is a demonstrator — keep a real extinguisher within reach.
