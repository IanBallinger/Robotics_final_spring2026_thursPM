#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "imu.h"
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"
#include "joystick.h"
#include "util.h"
#include "wireless.h"

struct DesiredWheelVel {
  float w1;
  float w2;
  float w3;
  float w4;

  DesiredWheelVel() : w1(0), w2(0), w3(0), w4(0) {}
  DesiredWheelVel(float w1_, float w2_, float w3_, float w4_)
      : w1(w1_), w2(w2_), w3(w3_), w4(w4_) {}
};

#define BNO08X_RESET 14
#define BNO08X_CS 12
#define BNO08X_INT 13
IMU imu(BNO08X_RESET, BNO08X_CS, BNO08X_INT);
String rx_line = "";

constexpr uint8_t num_wheels = 4;
constexpr float PID_TAU = 0.1f;
constexpr float ENCODER_SIGN[num_wheels] = {1.0f, -1.0f, 1.0f, -1.0f};

#ifndef MANUAL_JOYSTICK_X_PIN
#define MANUAL_JOYSTICK_X_PIN A0
#endif

#ifndef MANUAL_JOYSTICK_Y_PIN
#define MANUAL_JOYSTICK_Y_PIN A1
#endif

#ifndef AUTONOMY_TOGGLE_BUTTON_PIN
#define AUTONOMY_TOGGLE_BUTTON_PIN 14
#endif

constexpr float JOYSTICK_DEADBAND = 0.1f;
constexpr float JOYSTICK_MAX_FORWARD = 6.0f;
constexpr float JOYSTICK_MAX_TURN = 3.0f;

// User-defined serial/control rates.
constexpr unsigned long CMD_APPLY_PERIOD_MS = 50;   // latest buffered wheel cmd -> motors
constexpr unsigned long ACK_PUBLISH_PERIOD_MS = 50; // latest applied wheel cmd -> host
constexpr unsigned long IMU_PUBLISH_PERIOD_MS = 50; // latest IMU sample -> host
constexpr unsigned long CMD_TIMEOUT_MS = 250;       // stop motors if host goes silent
constexpr unsigned long JOYSTICK_APPLY_PERIOD_MS = 50;
constexpr unsigned long BUTTON_DEBOUNCE_MS = 50;
constexpr unsigned long CONTROLLER_TIMEOUT_MS = 250;
constexpr float WHEEL_CMD_FILTER_TAU_S = 0.03f;
constexpr float IMU_ACCEL_FILTER_TAU_S = 0.5f;
constexpr bool SERIAL_DEBUG_TIMING = true;

MotorDriver wheels[num_wheels] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3}
};

float kp[num_wheels] = {0.4f, 0.38f, 0.4f, 0.4f};
float ki[num_wheels] = {0.5f, 0.7f, 0.7f, 0.5f};
float kd[num_wheels] = {0.0f, 0.0f, 0.0f, 0.0f};

EncoderVelocity encoders[num_wheels] = {
    {ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER3_A_PIN, ENCODER3_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER4_A_PIN, ENCODER4_B_PIN, CPR_312_RPM, 0.2},
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

DesiredWheelVel latest_rx_cmd;
DesiredWheelVel latest_applied_cmd;
unsigned long last_wheel_cmd_filter_ms = 0;
bool has_pending_cmd = false;
bool ack_dirty = false;

Joystick manual_joystick(MANUAL_JOYSTICK_X_PIN, MANUAL_JOYSTICK_Y_PIN);
bool autonomy_enabled = false;
bool last_button_level = HIGH;
unsigned long last_button_change_ms = 0;
unsigned long last_joystick_apply_ms = 0;
unsigned long last_controller_rx_ms = 0;

unsigned long last_cmd_rx_ms = 0;
unsigned long last_cmd_apply_ms = 0;
unsigned long last_ack_publish_ms = 0;
unsigned long last_imu_publish_ms = 0;

unsigned long last_ack_debug_ms = 0;
unsigned long last_imu_debug_ms = 0;
unsigned long last_accel_filter_ms = 0;

float filtered_ax = 0.0f;
float filtered_ay = 0.0f;
float filtered_az = 0.0f;

const uint8_t* peerAddr = controllerAddr;
esp_now_peer_info_t peerInfo;

bool freshWirelessData = false;
ControllerMessage controllerMessage;
RobotMessage robotMessage;

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

bool sendRobotData() {
  esp_err_t result = esp_now_send(controllerAddr, (uint8_t*)&robotMessage,
                                  sizeof(robotMessage));
  return result == ESP_OK;
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

  // Match physical wheel polarity:
  // w1 = left_front, w2 = right_front, w3 = left_rear, w4 = right_rear.
  des_wheel_spd.w1 *= -1.0f;
  des_wheel_spd.w2 *= -1.0f;
  des_wheel_spd.w4 *= -1.0f;

  return true;
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
    wheels[i].drive(0.0);
  }
}

static DesiredWheelVel lowPassWheelCommand(const DesiredWheelVel& target,
                                           unsigned long now_ms) {
  if (last_wheel_cmd_filter_ms == 0) {
    last_wheel_cmd_filter_ms = now_ms;
    latest_applied_cmd = target;
    return latest_applied_cmd;
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

static void applyWheelCommand(const DesiredWheelVel& cmd) {
  const float setpoints[num_wheels] = {cmd.w1, cmd.w2, cmd.w3, cmd.w4};

  for (uint8_t i = 0; i < num_wheels; ++i) {
    measured_vel[i] = ENCODER_SIGN[i] * encoders[i].getVelocity();
    control_effort[i] = pids[i].calculateParallel(measured_vel[i], setpoints[i]);
    wheels[i].drive(control_effort[i]);
  }
}

static bool joystickToWheelCommand(const ControllerMessage& controller_msg,
                                  DesiredWheelVel& des_wheel_spd) {
  const float forward_input = controller_msg.joystick2.y;
  const float turn_input = controller_msg.joystick1.y;

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

  Serial.print("JOY_CMD,forward,");
  Serial.print(forward);
  Serial.print(",turn,");
  Serial.println(turn);

  // Differential/skid-steer mixing:
  //   joystick1.y -> forward/back
  //   joystick2.y -> turn in place
  // Wheel command order here is:
  //   w1 = left_rear
  //   w2 = left_front
  //   w3 = right_front
  //   w4 = right_rear
  const float left = forward - turn;
  const float right = forward + turn;

  String wheel_cmd_line = "WHL_CMD,";
  wheel_cmd_line += String(-left, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(left, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(right, 4);
  wheel_cmd_line += ",";
  wheel_cmd_line += String(right, 4);

  Serial.println(wheel_cmd_line);

  return handleWheelCommand(wheel_cmd_line, des_wheel_spd);
}

static void updateAutonomyToggle() {
  const unsigned long now = millis();
  const bool button_level = digitalRead(AUTONOMY_TOGGLE_BUTTON_PIN);

  if (button_level != last_button_level &&
      now - last_button_change_ms >= BUTTON_DEBOUNCE_MS) {
    last_button_change_ms = now;
    last_button_level = button_level;

    if (button_level == LOW) {
      autonomy_enabled = !autonomy_enabled;
      stopMotors();
      Serial.print("MODE,");
      Serial.println(autonomy_enabled ? "AUTONOMY" : "JOYSTICK");
    }
  }
}

static void printWheelAck(const DesiredWheelVel& cmd) {
  printDebugTiming("ACK", last_ack_debug_ms);

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

void sendIMU() {
  // Latest fused samples from BNO08x (imu.update() is called each loop).
  AccelReadings a = imu.getAccelReadings();
  GyroReadings g = imu.getGyroReadings();

  double ax_bias = 0.4f;

  const unsigned long now = millis();
  float dt = 0.0f;
  if (last_accel_filter_ms == 0) {
    filtered_ax = static_cast<float>(a.ax) + ax_bias;
    filtered_ay = static_cast<float>(a.ay);
    filtered_az = static_cast<float>(a.az);
  } else {
    dt = (now - last_accel_filter_ms) / 1000.0f;
    const float alpha = dt / (IMU_ACCEL_FILTER_TAU_S + dt);
    filtered_ax += alpha * (static_cast<float>(a.ax) + ax_bias - filtered_ax);
    filtered_ay += alpha * (static_cast<float>(a.ay) - filtered_ay);
    filtered_az += alpha * (static_cast<float>(a.az) - filtered_az);
  }
  last_accel_filter_ms = now;

  printDebugTiming("IMU", last_imu_debug_ms);
  Serial.print("IMU,");
  Serial.print(filtered_ax);
  Serial.print(",");
  Serial.print(filtered_ay);
  Serial.print(",");
  Serial.print(filtered_az);
  Serial.print(",");
  Serial.print(static_cast<float>(g.rollRate));
  Serial.print(",");
  Serial.print(static_cast<float>(g.pitchRate));
  Serial.print(",");
  Serial.println(static_cast<float>(g.yawRate));
}

void setup() {
  Serial.begin(115200);
  imu.setup();

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
    pids[i].setParallelTunings(kp[i], ki[i], kd[i], PID_TAU, integral_min,
                               integral_max);
  }

  manual_joystick.setup();
  pinMode(AUTONOMY_TOGGLE_BUTTON_PIN, INPUT_PULLUP);
  setupWireless();

  const unsigned long now = millis();
  last_cmd_rx_ms = now;
  last_cmd_apply_ms = now;
  last_ack_publish_ms = now;
  last_imu_publish_ms = now;
  last_button_change_ms = now;
  last_joystick_apply_ms = now;
  last_controller_rx_ms = now;
  last_wheel_cmd_filter_ms = now;
}

void loop() {
  imu.update();

  updateAutonomyToggle();

  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      rx_line.trim();

      DesiredWheelVel cmd;
      if (rx_line.length() > 0 && handleWheelCommand(rx_line, cmd)) {
        latest_rx_cmd = cmd;
        has_pending_cmd = true;
        last_cmd_rx_ms = millis();
      }

      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();

  if (autonomy_enabled) {
    if (now - last_cmd_rx_ms > CMD_TIMEOUT_MS) {
      stopMotors();
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

  // if (ack_dirty && (now - last_ack_publish_ms >= ACK_PUBLISH_PERIOD_MS)) {
  //   printWheelAck(latest_applied_cmd);
  //   ack_dirty = false;
  //   last_ack_publish_ms = now;
  // }

  if (now - last_imu_publish_ms >= IMU_PUBLISH_PERIOD_MS) {
    sendIMU();
    last_imu_publish_ms = now;
  }
}
