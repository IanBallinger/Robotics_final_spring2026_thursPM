#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <math.h>
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"
#include "util.h"
#include "wireless.h"

// Host serial protocol handled here:
//   MODE,AUTONOMY
//   MODE,JOYSTICK
//   WHL_CMD,w1,w2,w3,w4
//
// Canonical wheel command order:
//   w1 = left_front, w2 = right_front, w3 = left_rear, w4 = right_rear
//
// This firmware is intentionally simplified for wheel-only autonomy bring-up:
// - no IMU publishing
// - robust serial line handling with overflow protection
// - explicit command timeout in autonomy mode
// - conservative command clamping / validation

struct DesiredWheelVel {
  float w1;
  float w2;
  float w3;
  float w4;

  DesiredWheelVel() : w1(0), w2(0), w3(0), w4(0) {}
  DesiredWheelVel(float w1_, float w2_, float w3_, float w4_)
      : w1(w1_), w2(w2_), w3(w3_), w4(w4_) {}
};

String rx_line = "";

constexpr uint8_t num_wheels = 4;
constexpr float PID_TAU = 0.1f;
constexpr float ENCODER_SIGN[num_wheels] = {1.0f, -1.0f, 1.0f, -1.0f};
constexpr size_t MAX_RX_LINE_LEN = 96;
constexpr float MAX_WHEEL_CMD_RAD_S = 3.0f;

#ifndef AUTONOMY_TOGGLE_BUTTON_PIN
#define AUTONOMY_TOGGLE_BUTTON_PIN 14
#endif

constexpr float JOYSTICK_DEADBAND = 0.1f;
constexpr float JOYSTICK_MAX_FORWARD = 3.0f;
constexpr float JOYSTICK_MAX_TURN = 3.0f;

// User-defined serial/control rates.
constexpr unsigned long CMD_APPLY_PERIOD_MS = 10;   // latest buffered wheel cmd -> motors
constexpr unsigned long ACK_PUBLISH_PERIOD_MS = 20; // latest applied wheel cmd -> host
constexpr unsigned long CMD_TIMEOUT_MS = 250;       // stop motors if host goes silent
constexpr unsigned long JOYSTICK_APPLY_PERIOD_MS = 50;
constexpr unsigned long BUTTON_DEBOUNCE_MS = 50;
constexpr unsigned long CONTROLLER_TIMEOUT_MS = 250;
constexpr float WHEEL_CMD_FILTER_TAU_S = 0.1f;
constexpr bool SERIAL_DEBUG_TIMING = false;

MotorDriver wheels[num_wheels] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3}
};

float kp[num_wheels] = {0.3f, 0.3f, 0.3f, 0.3f};
float ki[num_wheels] = {0.05f, 0.05f, 0.05f, 0.05f}; // old integral: 0.4f
float kd[num_wheels] = {0.01f, 0.01f, 0.01f, 0.01f};

EncoderVelocity encoders[num_wheels] = {
    {ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.5},
    {ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.5},
    {ENCODER3_A_PIN, ENCODER3_B_PIN, CPR_312_RPM, 0.5},
    {ENCODER4_A_PIN, ENCODER4_B_PIN, CPR_312_RPM, 0.5},
};

PID pid1 = {kp[0], ki[0], kd[0], 0.0, PID_TAU, false};
PID pid2 = {kp[1], ki[1], kd[1], 0.0, PID_TAU, false};
PID pid3 = {kp[2], ki[2], kd[2], 0.0, PID_TAU, false};
PID pid4 = {kp[3], ki[3], kd[3], 0.0, PID_TAU, false};
PID pids[num_wheels] = {pid1, pid2, pid3, pid4};

double integral_min = -1e6;
double integral_max = 1e6;

double measured_vel[num_wheels] = {0.0, 0.0, 0.0, 0.0};
double control_effort[num_wheels] = {0.0, 0.0, 0.0, 0.0};

constexpr float ENCODER_MAX_VALID_SPEED_RAD_S = 10.0f;
constexpr float ENCODER_VEL_JUMP_MARGIN_RAD_S = 2.0f;
bool validated_encoder_state_initialized[num_wheels] = {false, false, false, false};
float validated_encoder_velocity_rad_s[num_wheels] = {0.0f, 0.0f, 0.0f, 0.0f};

DesiredWheelVel latest_rx_cmd;
DesiredWheelVel latest_applied_cmd;
unsigned long last_wheel_cmd_filter_ms = 0;
bool has_pending_cmd = false;
bool ack_dirty = false;

bool autonomy_enabled = false;
bool last_button_level = HIGH;
unsigned long last_button_change_ms = 0;
unsigned long last_joystick_apply_ms = 0;
unsigned long last_controller_rx_ms = 0;

// Remote (ESP-NOW): BUTTON_R on controller toggles manual <-> autonomy on press edge.
static bool prev_remote_mode_button = false;
static unsigned long last_remote_toggle_ms = 0;

unsigned long last_cmd_rx_ms = 0;
unsigned long last_cmd_apply_ms = 0;
unsigned long last_ack_publish_ms = 0;

unsigned long last_ack_debug_ms = 0;

const uint8_t* peerAddr = controllerAddr;
esp_now_peer_info_t peerInfo = {};
bool freshWirelessData = false;
ControllerMessage controllerMessage = {};
RobotMessage robotMessage = {};

static void stopMotors();

void onSendData(const uint8_t* mac_addr, esp_now_send_status_t status) {
  if (Serial) {
    Serial.print("ESP_NOW_SEND,");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
  }
}

void onRecvData(const uint8_t* mac, const uint8_t* incomingData, int len) {
  if (len != sizeof(ControllerMessage)) {
    if (Serial) {
      Serial.print("ESP_NOW_RX_BAD_LEN,");
      Serial.println(len);
    }
    return;
  }

  memcpy(&controllerMessage, incomingData, sizeof(controllerMessage));
  freshWirelessData = true;
  last_controller_rx_ms = millis();
}

static bool isFiniteWheelCommand(const DesiredWheelVel& cmd) {
  return isfinite(cmd.w1) && isfinite(cmd.w2) && isfinite(cmd.w3) && isfinite(cmd.w4);
}

static void clampWheelCommand(DesiredWheelVel& cmd) {
  cmd.w1 = constrain(cmd.w1, -MAX_WHEEL_CMD_RAD_S, MAX_WHEEL_CMD_RAD_S);
  cmd.w2 = constrain(cmd.w2, -MAX_WHEEL_CMD_RAD_S, MAX_WHEEL_CMD_RAD_S);
  cmd.w3 = constrain(cmd.w3, -MAX_WHEEL_CMD_RAD_S, MAX_WHEEL_CMD_RAD_S);
  cmd.w4 = constrain(cmd.w4, -MAX_WHEEL_CMD_RAD_S, MAX_WHEEL_CMD_RAD_S);
}

bool handleWheelCommand(const String& line, DesiredWheelVel& des_wheel_spd) {
  if (!line.startsWith("WHL_CMD,")) {
    Serial.println("WRONG_START");
    return false;
  }

  int parsed = sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &des_wheel_spd.w1,
                      &des_wheel_spd.w2, &des_wheel_spd.w3, &des_wheel_spd.w4);
  if (parsed != 4) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  if (!isFiniteWheelCommand(des_wheel_spd)) {
    Serial.println("ERR,NAN_CMD");
    return false;
  }

  clampWheelCommand(des_wheel_spd);

  // Match physical wheel polarity:
  // w1 = left_front, w2 = right_front, w3 = left_rear, w4 = right_rear.
  des_wheel_spd.w2 *= -1.0f;
  des_wheel_spd.w3 *= -1.0f;
  return true;
}

bool handleModeCommand(const String& line) {
  if (line == "MODE,AUTONOMY") {
    autonomy_enabled = true;
    stopMotors();
    Serial.println("MODE,AUTONOMY");
    return true;
  }
  if (line == "MODE,JOYSTICK") {
    autonomy_enabled = false;
    stopMotors();
    has_pending_cmd = false;
    Serial.println("MODE,JOYSTICK");
    return true;
  }
  return false;
}

static void printDebugTiming(const char* tag, unsigned long& last_ms) {
  if (!SERIAL_DEBUG_TIMING) {
    return;
  }

  const unsigned long now = millis();
  Serial.print("DBG,");
  Serial.print(tag);
  Serial.print(",dt_ms,");
  if (last_ms == 0) {
    Serial.println("FIRST");
  } else {
    Serial.println(now - last_ms);
  }
  last_ms = now;
}

static void stopMotors() {
  latest_rx_cmd = DesiredWheelVel();
  latest_applied_cmd = DesiredWheelVel();
  last_wheel_cmd_filter_ms = 0;
  has_pending_cmd = false;
  ack_dirty = false;

  for (uint8_t i = 0; i < num_wheels; ++i) {
    control_effort[i] = 0.0;
    measured_vel[i] = 0.0;
    validated_encoder_velocity_rad_s[i] = 0.0f;
    validated_encoder_state_initialized[i] = false;
    wheels[i].drive(0.0);
  }
}

static DesiredWheelVel lowPassWheelCommand(const DesiredWheelVel& target,
                                           unsigned long now_ms) {
  if (last_wheel_cmd_filter_ms == 0) {
    last_wheel_cmd_filter_ms = now_ms;
  }

  const float dt = (now_ms - last_wheel_cmd_filter_ms) / 1000.0f;
  const float alpha = dt / (WHEEL_CMD_FILTER_TAU_S + dt);

  latest_applied_cmd.w1 += alpha * (target.w1 - latest_applied_cmd.w1);
  latest_applied_cmd.w2 += alpha * (target.w2 - latest_applied_cmd.w2);
  latest_applied_cmd.w3 += alpha * (target.w3 - latest_applied_cmd.w3);
  latest_applied_cmd.w4 += alpha * (target.w4 - latest_applied_cmd.w4);

  last_wheel_cmd_filter_ms = now_ms;
  return latest_applied_cmd;
}

static float readValidatedWheelVelocity(uint8_t wheel_index) {
  const float raw_velocity_rad_s =
      ENCODER_SIGN[wheel_index] * encoders[wheel_index].getVelocity();

  if (!validated_encoder_state_initialized[wheel_index]) {
    validated_encoder_velocity_rad_s[wheel_index] = 0.0f;
    validated_encoder_state_initialized[wheel_index] = true;
  }

  const float max_allowed_speed =
      ENCODER_MAX_VALID_SPEED_RAD_S + ENCODER_VEL_JUMP_MARGIN_RAD_S;
  if (fabsf(raw_velocity_rad_s) <= max_allowed_speed) {
    validated_encoder_velocity_rad_s[wheel_index] = raw_velocity_rad_s;
  }

  return validated_encoder_velocity_rad_s[wheel_index];
}

static void applyWheelCommand(const DesiredWheelVel& cmd) {
  const float setpoints[num_wheels] = {cmd.w1, cmd.w2, cmd.w3, cmd.w4};

  for (uint8_t i = 0; i < num_wheels; ++i) {
    measured_vel[i] = readValidatedWheelVelocity(i);

    if (setpoints[i] == 0.0f) {
      control_effort[i] = 0.0;
      measured_vel[i] = 0.0;
      validated_encoder_velocity_rad_s[i] = 0.0f;
      validated_encoder_state_initialized[i] = false;
      wheels[i].drive(0.0);
    } else {
      control_effort[i] = pids[i].calculateParallel(measured_vel[i], setpoints[i]);
      wheels[i].drive(control_effort[i]);
    }
  }
}

static bool joystickToWheelCommand(const ControllerMessage& controller_msg,
                                  DesiredWheelVel& des_wheel_spd) {
  const float forward_input = controller_msg.joystick1.y;
  const float turn_input = controller_msg.joystick2.x;

  const float forward = fabs(forward_input) < JOYSTICK_DEADBAND
                            ? 0.0f
                            : static_cast<float>(mapDouble(forward_input, -1.0, 1.0,
                                                           -JOYSTICK_MAX_FORWARD,
                                                           JOYSTICK_MAX_FORWARD));
  const float turn = fabs(turn_input) < JOYSTICK_DEADBAND
                         ? 0.0f
                         : static_cast<float>(mapDouble(turn_input, -1.0, 1.0,
                                                        -JOYSTICK_MAX_TURN,
                                                        JOYSTICK_MAX_TURN));

  // Differential/skid-steer mixing:
  //   joystick1.y -> forward/back
  //   joystick2.x -> turn in place
  const float left = forward + turn;
  const float right = forward - turn;

  String wheel_cmd_line = "WHL_CMD,";
  wheel_cmd_line += String(left, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(left, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(right, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(right, 4);

  Serial.println(wheel_cmd_line);

  return handleWheelCommand(wheel_cmd_line, des_wheel_spd);
}

static void toggleAutonomyMode() {
  autonomy_enabled = !autonomy_enabled;
  stopMotors();
  Serial.print("MODE,");
  Serial.println(autonomy_enabled ? "AUTONOMY" : "JOYSTICK");
}

static void updateAutonomyToggle() {
  const unsigned long now = millis();
  const bool button_level = digitalRead(AUTONOMY_TOGGLE_BUTTON_PIN);

  if (button_level != last_button_level &&
      now - last_button_change_ms >= BUTTON_DEBOUNCE_MS) {
    last_button_change_ms = now;
    last_button_level = button_level;

    if (button_level == LOW) {
      toggleAutonomyMode();
    }
  }
}

static void updateRemoteAutonomyToggle() {
  const unsigned long now = millis();
  constexpr unsigned long REMOTE_TOGGLE_DEBOUNCE_MS = 400;

  const bool pressed = controllerMessage.buttonR;
  if (pressed && !prev_remote_mode_button &&
      now - last_remote_toggle_ms >= REMOTE_TOGGLE_DEBOUNCE_MS) {
    toggleAutonomyMode();
    last_remote_toggle_ms = now;
  }
  prev_remote_mode_button = pressed;
}

static void printWheelAck(const DesiredWheelVel& cmd) {
  printDebugTiming("ACK", last_ack_debug_ms);

  Serial.println("DBG,WHEEL_ORDER,w1_left_front,w2_right_front,w3_left_rear,w4_right_rear");
  Serial.print("CMD,");
  Serial.print(cmd.w1);
  Serial.print(",");
  Serial.print(cmd.w2);
  Serial.print(",");
  Serial.print(cmd.w3);
  Serial.print(",");
  Serial.println(cmd.w4);

  Serial.print("ENC,");
  Serial.print(measured_vel[0]);
  Serial.print(",");
  Serial.print(measured_vel[1]);
  Serial.print(",");
  Serial.print(measured_vel[2]);
  Serial.print(",");
  Serial.println(measured_vel[3]);

  Serial.print("EFF,");
  Serial.print(control_effort[0]);
  Serial.print(",");
  Serial.print(control_effort[1]);
  Serial.print(",");
  Serial.print(control_effort[2]);
  Serial.print(",");
  Serial.println(control_effort[3]);
}

static void processSerialLine(const String& line) {
  DesiredWheelVel cmd;
  if (line.length() == 0) {
    return;
  }
  if (handleModeCommand(line)) {
    return;
  }
  if (!handleWheelCommand(line, cmd)) {
    return;
  }

  latest_rx_cmd = cmd;
  has_pending_cmd = true;
  last_cmd_rx_ms = millis();

  if (!autonomy_enabled) {
    autonomy_enabled = true;
    stopMotors();
    latest_rx_cmd = cmd;
    has_pending_cmd = true;
    last_cmd_rx_ms = millis();
    Serial.println("MODE,AUTONOMY_SERIAL");
  }
}

void setup() {
  Serial.begin(115200);
  rx_line.reserve(MAX_RX_LINE_LEN);

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
    pids[i].setParallelTunings(kp[i], ki[i], kd[i], PID_TAU, integral_min,
                               integral_max);
  }

  pinMode(AUTONOMY_TOGGLE_BUTTON_PIN, INPUT_PULLUP);
  setupWireless();

  const unsigned long now = millis();
  last_cmd_rx_ms = now;
  last_cmd_apply_ms = now;
  last_ack_publish_ms = now;
  last_button_change_ms = now;
  last_joystick_apply_ms = now;
  last_controller_rx_ms = now;
  last_wheel_cmd_filter_ms = now;
}

void loop() {
  updateAutonomyToggle();
  updateRemoteAutonomyToggle();

  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      rx_line.trim();
      processSerialLine(rx_line);
      rx_line = "";
      continue;
    }

    if (rx_line.length() >= MAX_RX_LINE_LEN) {
      rx_line = "";
      Serial.println("ERR,RX_OVERFLOW");
      continue;
    }
    rx_line += c;
  }

  const unsigned long now = millis();

  if (autonomy_enabled) {
    if (now - last_cmd_rx_ms > CMD_TIMEOUT_MS) {
      if (latest_rx_cmd.w1 != 0.0f || latest_rx_cmd.w2 != 0.0f ||
          latest_rx_cmd.w3 != 0.0f || latest_rx_cmd.w4 != 0.0f) {
        Serial.println("ERR,CMD_TIMEOUT");
      }
      latest_rx_cmd = DesiredWheelVel();
      has_pending_cmd = false;
    }

    if (now - last_cmd_apply_ms >= CMD_APPLY_PERIOD_MS) {
      applyWheelCommand(lowPassWheelCommand(latest_rx_cmd, now));
      ack_dirty = true;
      has_pending_cmd = false;
      last_cmd_apply_ms = now;
    }
  } else if (now - last_joystick_apply_ms >= JOYSTICK_APPLY_PERIOD_MS) {
    if (now - last_controller_rx_ms > CONTROLLER_TIMEOUT_MS) {
      stopMotors();
      Serial.println("ERR,CONTROLLER_TIMEOUT");
    } else {
      DesiredWheelVel joystick_cmd;
      if (joystickToWheelCommand(controllerMessage, joystick_cmd)) {
        latest_rx_cmd = joystick_cmd;
        applyWheelCommand(lowPassWheelCommand(latest_rx_cmd, now));
        ack_dirty = true;
      } else {
        Serial.println("ERR,BAD_JOYSTICK_CMD");
      }
    }
    last_joystick_apply_ms = now;
  }

  if (ack_dirty && (now - last_ack_publish_ms >= ACK_PUBLISH_PERIOD_MS)) {
    printWheelAck(latest_applied_cmd);
    ack_dirty = false;
    last_ack_publish_ms = now;
  }

}
