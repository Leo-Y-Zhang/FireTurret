# Running FireTurret on a Raspberry Pi

The goal: plug a camera into the Pi, run one command, and view/control the
turret from your phone. Everything below is on the Pi itself.

## 1. Camera

Either works:

- **Raspberry Pi Camera Module** (CSI ribbon) — enable it once:
  `sudo raspi-config` → *Interface Options* → *Camera* → enable, then reboot.
  It appears to OpenCV as index `0`.
- **USB webcam** — just plug it in; it's index `0` (or `1` if the Pi camera is
  also present). Check with `ls /dev/video*`.

## 2. Install

```bash
sudo apt update && sudo apt install -y python3-venv libgl1 libglib2.0-0
git clone <your FireTurret repo>  &&  cd FireTurret
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

## 3. Run

**Browser interface (recommended for a mounted turret).** Serve a live view,
operator warnings, and a remote E-stop; open it from any phone/laptop on the
same network:

```bash
python -m fireturret web --source 0                 # camera 0, dry-run (no turret) — LOOPBACK only
python -m fireturret web --source 0 --host 0.0.0.0 --expose         # reachable from your phone (read the warning below)
python -m fireturret web --source 0 --port /dev/ttyUSB0             # pan/tilt track, water OFF (dry-aim)
python -m fireturret web --source 0 --port /dev/ttyUSB0 --arm-water # + enable water (asks you to type ARM)
```

Then browse to `http://<pi-address>:8000` (find the address with `hostname -I`,
or use `http://raspberrypi.local:8000`). No hardware? Try `--source sim`.

**Why `--expose` exists.** The console is **unauthenticated**, and its `POST
/estop` latches the mission into SAFE and disarms water with no rearm path from
the browser. Bound to a routable address, anything that reaches that network can
irreversibly disable the turret — and watch the camera. So it binds to loopback
unless you say otherwise. On a home LAN that is usually acceptable for a
supervised demo; on anything less trusted, put a VPN or an authenticating reverse
proxy in front rather than exposing it directly.

**Local window** (if the Pi has a display): `python -m fireturret run --source 0`.

**Dry-run first.** Without `--port` nothing is actuated — you get the detection
and aiming overlay only. Even *with* `--port`, water stays OFF (dry-aim: pan/tilt
track the fire but the pump/valve are disabled) unless you add `--arm-water` and
type `ARM` to confirm — the same safe default as `fireturret run`. Always confirm
detection and the aim solution look right before arming water. Then calibrate per
`BUILD_GUIDE.md` §5.

## 4. Autostart on boot (optional)

Create `/etc/systemd/system/fire-turret.service`:

```ini
[Unit]
Description=FireTurret camera-guided fire-suppression turret
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/FireTurret
ExecStart=/home/pi/FireTurret/.venv/bin/python -m fireturret web --source 0 --port /dev/ttyUSB0 --host 0.0.0.0 --expose
Restart=on-failure
# NOTE: this autostart runs DRY-AIM (water off) by design — there is no terminal
# to type ARM at, so water is never armed unattended. Arm water only from an
# interactive `fireturret web ... --arm-water` session with a supervisor present.
# NOTE: --expose is required because the unit binds a routable address so you can
# reach the console from a phone. That console is unauthenticated and its E-stop
# is irreversible from the browser, so only do this on a network you trust — an
# unattended, permanently-listening turret is the strongest case for a VPN or an
# authenticating reverse proxy. Drop --host/--expose to keep it loopback-only.

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now fire-turret
```

The turret now starts on power-up and the interface is always at
`http://<pi-address>:8000`. The firmware's own E-stop and heartbeat watchdog
still protect the hardware independently of any of this.

## Performance

The pipeline is classical CV (~125 fps on a desktop; a Pi 5 is slower but real
time at this resolution). If you need more headroom, drop `CameraConfig.width`/
`height` in `src/fireturret/config.py` and re-run — the control loop is
resolution-independent.
