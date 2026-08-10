/*
 * FireTurret firmware — Arduino Uno / Nano / ESP32.
 *
 * Speaks the line protocol in src/fireturret/rig/protocol.py (keep in lockstep):
 *   Host -> board:  C p=<pan> t=<tilt> w=<pump%> v=<0|1> l=<0|1> x=<0|1>\n  (also heartbeat)
 *   board -> host:  S p=<pan> t=<tilt> w=<pump%> v=<0|1> e=<0|1> h=<0|1>\n  (10 Hz)
 * (l = laser accessory, x = operator-warning indicator, e = E-stop engaged,
 *  h = pan axis homed.)
 *
 * SAFETY (independent of the host):
 *  - Hardware E-stop on E_STOP_PIN (active-low, normally-closed): forces pump
 *    and valve off and freezes motion while engaged.
 *  - Heartbeat watchdog: if no valid command for HEARTBEAT_TIMEOUT_MS, the
 *    pump and valve are cut. The host must keep sending commands.
 *  - Homing: the pan angle is dead-reckoned, so pan motion and water are LOCKED
 *    until the pan axis has homed against its limit switch (a known reference).
 *    An un-homed turret cannot move pan or spray.
 *  - Pan/tilt are clamped to the mechanical limits below.
 *
 * Pan is a stepper via a STEP/DIR driver (TMC2209/A4988); tilt is a hobby
 * servo; the pump is PWM through a logic-level MOSFET; the valve is a MOSFET
 * or relay. Adjust pins and limits to your build (see docs/BUILD_GUIDE.md).
 */

#include <Servo.h>

// ---- pins ----
const int PAN_STEP_PIN = 2;
const int PAN_DIR_PIN = 5;
const int TILT_SERVO_PIN = 9;
const int PUMP_PWM_PIN = 3;   // must be a PWM pin
const int VALVE_PIN = 7;
const int LASER_PIN = 8;
const int WARN_PIN = 6;       // operator warning LED/buzzer (drive from x=)
const int E_STOP_PIN = 4;     // active-low, INPUT_PULLUP, normally-closed
const int HOME_SWITCH_PIN = 10; // pan home limit switch, active-low (INPUT_PULLUP)

// ---- mechanical limits (keep in sync with TurretConfig) ----
const float PAN_MIN = -170.0, PAN_MAX = 170.0;
const float TILT_MIN = 8.0, TILT_MAX = 50.0;
const float PAN_STEPS_PER_DEG = 8.888; // 200 steps * 16 microsteps / 360
const unsigned long HEARTBEAT_TIMEOUT_MS = 500;
const unsigned long STEP_INTERVAL_US = 250; // pan slew speed cap

// ---- homing ----
const float PAN_HOME_DEG = PAN_MIN;   // angle assigned when the home switch triggers
const long HOME_MAX_STEPS = (long)((PAN_MAX - PAN_MIN) * PAN_STEPS_PER_DEG * 1.2);

Servo tiltServo;

float panTarget = 0, panActual = 0;
float tiltTarget = 20;
float pumpPct = 0;
bool valveOpen = false;
bool laserOn = false;
bool warnOn = false;
bool homed = false;           // pan reference established? (water + pan gated on this)

unsigned long lastCommandMs = 0;
unsigned long lastReportMs = 0;
unsigned long lastStepUs = 0;

// Longest line the host can emit is
//   "C p=-179.99 t=-179.99 w=-100.0 v=1 l=1 x=1\n"  = 43 bytes.
// Sized well clear of that; tests/test_safety_debt.py pins the host side under
// this number so a command can never be silently truncated.
char buf[96];
uint8_t bufLen = 0;
// Set when a line exceeds the buffer. A truncated line is DROPPED rather than
// parsed, because a truncated "close the valve" used to leave water flowing.
bool lineOverflowed = false;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

bool eStopEngaged() {
  return digitalRead(E_STOP_PIN) == LOW; // NC switch opens -> pulled LOW -> engaged
}

bool homeSwitchPressed() {
  return digitalRead(HOME_SWITCH_PIN) == LOW; // active-low
}

// Gentle, supervised homing: step toward the home switch until it triggers, then
// adopt a known reference angle. Without a switch this times out and leaves the
// turret UN-HOMED, so pan motion and water stay locked (see stepPanToward /
// applyOutputs). Runs once at boot; the operator must be watching.
void homePan() {
  if (eStopEngaged()) return;          // never move under E-stop
  digitalWrite(PAN_DIR_PIN, LOW);      // sweep toward PAN_MIN / the switch
  for (long i = 0; i < HOME_MAX_STEPS; i++) {
    if (eStopEngaged()) return;
    if (homeSwitchPressed()) {
      panActual = PAN_HOME_DEG;
      panTarget = 0.0;                 // then re-centre under normal control
      homed = true;
      return;
    }
    digitalWrite(PAN_STEP_PIN, HIGH);
    delayMicroseconds(3);
    digitalWrite(PAN_STEP_PIN, LOW);
    delayMicroseconds(600);            // slow homing speed
  }
  // no switch found within travel: stay un-homed (water + pan remain locked)
}

void applyOutputs() {
  // warning indicator is driven regardless of failsafe — E-stop / signal loss
  // is exactly when the operator needs to see it
  digitalWrite(WARN_PIN, warnOn ? HIGH : LOW);
  bool linkOk = !eStopEngaged() &&
                (millis() - lastCommandMs) <= HEARTBEAT_TIMEOUT_MS;
  // water additionally requires a homed pan axis (a known reference)
  bool waterOk = linkOk && homed;
  if (!waterOk) {        // failsafe: kill pump + valve
    analogWrite(PUMP_PWM_PIN, 0);
    digitalWrite(VALVE_PIN, LOW);
  } else {
    analogWrite(PUMP_PWM_PIN, (int)(clampf(pumpPct, 0, 100) * 2.55));
    digitalWrite(VALVE_PIN, valveOpen ? HIGH : LOW);
  }
  digitalWrite(LASER_PIN, laserOn ? HIGH : LOW);
  if (linkOk) tiltServo.write((int)clampf(tiltTarget, TILT_MIN, TILT_MAX));
}

void stepPanToward() {
  if (eStopEngaged()) return;
  if (!homed) return;  // no pan motion until the axis has a known reference
  if (micros() - lastStepUs < STEP_INTERVAL_US) return;
  float target = clampf(panTarget, PAN_MIN, PAN_MAX);
  float diff = target - panActual;
  if (fabs(diff) < 1.0 / PAN_STEPS_PER_DEG) return;
  digitalWrite(PAN_DIR_PIN, diff > 0 ? HIGH : LOW);
  digitalWrite(PAN_STEP_PIN, HIGH);
  delayMicroseconds(3);
  digitalWrite(PAN_STEP_PIN, LOW);
  panActual += (diff > 0 ? 1.0 : -1.0) / PAN_STEPS_PER_DEG;
  lastStepUs = micros();
}

// Match a whole key up to '=' — NOT just the first character. Dispatching on
// tok[0] means any future token beginning p/t/w/v/l/x silently overwrites an
// existing field, which is how a protocol extension turns into a wrong pump
// setting on an unpatched board.
static bool keyIs(const char *tok, const char *key) {
  size_t n = 0;
  while (key[n] != '\0') {
    if (tok[n] != key[n]) return false;
    n++;
  }
  return tok[n] == '=';
}

// Returns false for a malformed line. The caller must NOT feed the watchdog on
// a false return: MALFORMED_LINE_DOES_NOT_FEED_WATCHDOG.
bool parseLine(char *line) {
  if (line[0] != 'C') return false;

  // Parse locals start from SAFE values and are NEVER seeded from the previous
  // command. The old code wrote `float p = panTarget, ... int v = valveOpen`, so
  // a line truncated before its v= token inherited the previous valve state —
  // a truncated "close the valve" left water flowing. Nothing here inherits.
  float p = 0.0f, t = TILT_MIN;   // safe aim; only used once sawPan/sawTilt hold
  float w = 0.0f;                 // pump: absent means OFF
  int v = 0, l = 0, x = 0;        // valve/laser/warn: absent means OFF
  bool sawPan = false, sawTilt = false, sawPump = false, sawValve = false;

  char *tok = strtok(line + 1, " ");
  while (tok) {
    if (keyIs(tok, "p")) { p = atof(tok + 2); sawPan = true; }
    else if (keyIs(tok, "t")) { t = atof(tok + 2); sawTilt = true; }
    else if (keyIs(tok, "w")) { w = atof(tok + 2); sawPump = true; }
    else if (keyIs(tok, "v")) { v = atoi(tok + 2); sawValve = true; }
    else if (keyIs(tok, "l")) l = atoi(tok + 2);
    else if (keyIs(tok, "x")) x = atoi(tok + 2);
    else return false;  // unknown token: treat the whole line as malformed
    tok = strtok(NULL, " ");
  }

  // A well-formed command carries all four control fields (the host encoder
  // always emits them, in this order). Any absence means the line was truncated
  // mid-flight, so refuse the whole line rather than acting on half of it.
  // Refusing — rather than applying safe defaults — is what keeps the aim from
  // jumping to 0 deg on a corrupted byte.
  if (!sawPan || !sawTilt || !sawPump || !sawValve) return false;

  panTarget = clampf(p, PAN_MIN, PAN_MAX);
  tiltTarget = clampf(t, TILT_MIN, TILT_MAX);
  pumpPct = clampf(w, 0, 100);
  valveOpen = v != 0;
  laserOn = l != 0;
  warnOn = x != 0;
  lastCommandMs = millis();
  return true;
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (lineOverflowed) {
        // The line was longer than the buffer, so what we hold is a truncated
        // prefix. Acting on it is exactly the fail-dangerous case: drop it, and
        // crucially do NOT refresh lastCommandMs — a truncated line must let the
        // heartbeat watchdog fire rather than feeding it.
        lineOverflowed = false;
      } else {
        buf[bufLen] = '\0';
        parseLine(buf);  // a false return deliberately leaves the watchdog unfed
      }
      bufLen = 0;
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = c;
    } else {
      lineOverflowed = true;
    }
  }
}

void report() {
  if (millis() - lastReportMs < 100) return; // 10 Hz
  lastReportMs = millis();
  Serial.print("S p="); Serial.print(panActual, 2);
  Serial.print(" t="); Serial.print(tiltTarget, 2);
  Serial.print(" w="); Serial.print(pumpPct, 1);
  Serial.print(" v="); Serial.print(valveOpen ? 1 : 0);
  Serial.print(" e="); Serial.print(eStopEngaged() ? 1 : 0);
  Serial.print(" h="); Serial.print(homed ? 1 : 0);
  Serial.print('\n');
}

void setup() {
  Serial.begin(115200);
  pinMode(PAN_STEP_PIN, OUTPUT);
  pinMode(PAN_DIR_PIN, OUTPUT);
  pinMode(PUMP_PWM_PIN, OUTPUT);
  pinMode(VALVE_PIN, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);
  pinMode(WARN_PIN, OUTPUT);
  pinMode(E_STOP_PIN, INPUT_PULLUP);
  pinMode(HOME_SWITCH_PIN, INPUT_PULLUP);
  tiltServo.attach(TILT_SERVO_PIN);
  tiltServo.write((int)tiltTarget);
  lastCommandMs = millis();
  homePan();  // establish the pan reference before any motion/water is allowed
}

void loop() {
  readSerial();
  stepPanToward();
  applyOutputs();
  report();
}
