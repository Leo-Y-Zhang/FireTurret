# Theory of operation

How this turret turns a single camera and a water jet into a closed-loop
fire-suppression aimer, and why each part is shaped the way it is. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the code-level rationale and
[`STUDIO.md`](STUDIO.md) for the desktop workbench.

## 1. The governing idea: direction, not distance

A monocular camera measures the *bearing* to a fire (which pixel it occupies) but
not the *range* — and range is exactly what a ballistic aimer needs, because it
sets how far to throw the water. Estimating range from a single image is biased
and brittle. This design avoids it: it observes **both** the fire and the water splash
in the same frame and drives the splash onto the fire in image space. Every error
source — model error, drag, wind, and the boresight offset between camera and
nozzle — manifests as a splash-to-fire pixel gap and is servoed away.

## 2. Ballistics

The jet is modelled as a drag-decelerated projectile leaving the nozzle at a speed
set by pump pressure.

- **Pressure → exit speed** (`exit_velocity`): a discharge/velocity coefficient
  applied to the Bernoulli speed, `v = C · √(2·p/ρ)`, with `p` proportional to the
  pump percent and `ρ` the density of water.
- **Trajectory** (`simulate_arc`): RK4 integration of the 2-D point-mass equations
  with quadratic drag, `a = −g·ŷ − k·|v|·v`, from the nozzle height to the ground.
  A single breakup constant `k` stands in for jet dispersion.
- **Range is unimodal in elevation.** Under quadratic drag the maximum-range angle
  sits well below the vacuum 45° (≈30° for a water jet). A target below peak range
  therefore has up to two solutions; `solve_elevation` returns the **low arc**
  (flatter, shorter flight, less wind exposure) when it can reach, and a high-arc
  lob only for targets closer than the minimum-elevation range.
- **Range is controlled by pump at a fixed low tilt.** During suppression the tilt
  is held on the low arc and **pump** becomes the single range axis
  (`solve_pump`), because range is monotonic in pump at fixed elevation — clean to
  servo. Servoing range with tilt near the peak is impossible: `dRange/dTilt ≈ 0`
  there.
- **Reachable envelope** (`reach_bounds`): the shortest coherent shot (minimum
  pump, minimum elevation) to the maximum reach (full pump, peak elevation). A
  target outside it cannot be hit — that is physics, and the mission reports it
  rather than hosing indefinitely.

`ballistics.fit_jet` calibrates the two soft constants (`velocity_coeff`,
`drag_k`) from measured test shots by least-squares grid search; the closed loop
absorbs whatever error remains.

## 3. Perception

- **Fire detection** separates saturated, warm, **flickering** regions using colour
  gates plus a temporal per-pixel toggle rate. Flicker is what distinguishes a real
  flame from a static red object (a cloth, a warning light) — the `decoy` scenario
  and its test exercise exactly this.
- **Tracking** adds persistence and negative evidence so a target is only
  *confirmed* after several consistent hits, and survives brief misses. It is
  **motion-compensated**: when the camera pans, a world-fixed fire slides across
  the image by a predictable pixel delta, which is subtracted before association so
  acquisition does not thrash during a sweep.
- **Splash detection** finds the desaturated, moving water. It is separated from
  the fire by **colour** (fire is saturated, water is not) rather than by excluding
  the fire's bounding box — because the splash is most informative precisely when it
  lands *on* the fire, where a bbox exclusion would blank it.

## 4. The visual servo

Suppression closes two independent loops, both in image space:

- **Azimuth (pan)** drives the horizontal column gap between the splash and the
  fire to zero. This is only correctable because **the camera rides the pan stage**:
  panning changes the fire's bearing but not the splash's, so the pan that aligns
  their columns also points the nozzle at the fire — absorbing the boresight offset.
- **Range (pump)** drives the vertical row gap: a splash landing short appears below
  the fire, long appears above. Pump is trimmed toward closing that row error.

The loop runs an **observe-then-correct** cadence: hold the aim dead still, collect
splash observations (frame-differencing needs a still camera), take the median, and
apply **one** small bounded correction per cycle. Continuous high-gain correction on
noisy per-frame vision oscillates; this median-and-step discipline is what keeps it
stable. Range is always **approached from below** (open the pump short): a short
splash lands in front of the flame and stays visible, whereas a long splash lands
behind it and is occluded, starving the very feedback the loop needs.

## 5. Mission state machine

`SEARCH → ACQUIRE → RANGE → ALIGN → SUPPRESS → CONFIRM`, with `HOLD` for
unreachable targets and `SAFE` overriding everything on fault. Pan uses a
**move → settle → observe** discipline (commanding an absolute azimuth and only
re-measuring once settled), because servoing every frame off a moving-camera
detection oscillates. A pump that stays saturated-and-short for several cycles
means the target is out of reach; the mission drops to `HOLD` with exponential
backoff instead of wasting water, and raises an operator advisory.

## 6. Why the closed loop works: the model-vs-true gap

The simulator deliberately runs **perturbed** physics — a different discharge
coefficient, more drag, and an azimuth bias — that the controller does not know.
The controller plans its opening shot from the nominal model, so the first water
lands off-target; the loop then observes the splash and corrects. The end-to-end
test passes *only* because the closed loop cancels this model error, which is the
whole thesis. Studio makes the gap visible: the ballistics explorer draws
the model arc solid and the true (perturbed) arc dashed, and the perturbation
parameters are editable in the scenario so you can widen the gap and watch the loop
still converge.

## 7. Failsafes

The turret is a supervised demonstrator, not a life-safety device, and the software
keeps its failsafes unconditionally: a **heartbeat watchdog** cuts pump and valve if
commands stop arriving, an **aim geofence** clamps pan/tilt to the mechanical
limits, and an **E-stop** (local or via the web console) forces the mission to
`SAFE`. On real hardware the emergency stop cuts pump and motor power independently
of the software.
