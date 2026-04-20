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

IMU imu(BNO08X_RESET, BNO08X_CS, BNO08X_INT);
String rx_line = "";

constexpr uint8_t num_wheels = 4;

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
constexpr bool SERIAL_DEBUG_TIMING = true;

MotorDriver wheels[num_wheels] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3}
};

// PID from wheel velocities to motor driver control efforts
#define Kp 0.25
#define Ki 0.01
#define Kd 0
#define pidTau 0.1

EncoderVelocity encoder1(ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2);
EncoderVelocity encoder2(ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2);
EncoderVelocity encoder3(ENCODER3_A_PIN, ENCODER3_B_PIN, CPR_312_RPM, 0.2);
EncoderVelocity encoder4(ENCODER4_A_PIN, ENCODER4_B_PIN, CPR_312_RPM, 0.2);

PID pid1(Kp, Ki, Kd, 0, pidTau, false);
PID pid2(Kp, Ki, Kd, 0, pidTau, false);
PID pid3(Kp, Ki, Kd, 0, pidTau, false);
PID pid4(Kp, Ki, Kd, 0, pidTau, false);

double velocity1 = 0;
double velocity2 = 0;
double velocity3 = 0;
double velocity4 = 0;
double controlEffort1 = 0;
double controlEffort2 = 0;
double controlEffort3 = 0;
double controlEffort4 = 0;

DesiredWheelVel latest_rx_cmd;
DesiredWheelVel latest_applied_cmd;
bool has_pending_cmd = false;
bool ack_dirty = false;

Joystick manual_joystick(MANUAL_JOYSTICK_X_PIN, MANUAL_JOYSTICK_Y_PIN);
bool autonomy_enabled = true;
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
  has_pending_cmd = false;
  ack_dirty = false;

  controlEffort1 = 0;
  controlEffort2 = 0;
  controlEffort3 = 0;
  controlEffort4 = 0;

  wheels[0].drive(0.0);
  wheels[1].drive(0.0);
  wheels[2].drive(0.0);
  wheels[3].drive(0.0);
}

static void applyWheelCommand(const DesiredWheelVel& cmd) {
  velocity1 = encoder1.getVelocity();
  velocity2 = encoder2.getVelocity();
  velocity3 = encoder3.getVelocity();
  velocity4 = encoder4.getVelocity();
  controlEffort1 = pid1.calculateParallel(velocity1, cmd.w1);
  controlEffort2 = pid2.calculateParallel(velocity2, cmd.w2);
  controlEffort3 = pid3.calculateParallel(velocity3, cmd.w3);
  controlEffort4 = pid4.calculateParallel(velocity4, cmd.w4);

  wheels[0].drive(controlEffort1);
  wheels[1].drive(controlEffort2);
  wheels[2].drive(controlEffort3);
  wheels[3].drive(controlEffort4);
}

static DesiredWheelVel joystickToWheelCommand(const JoystickReading& reading) {
  const float forward = fabs(reading.y) < JOYSTICK_DEADBAND
                            ? 0.0f
                            : static_cast<float>(mapDouble(reading.y, -1.0, 1.0,
                                                           -JOYSTICK_MAX_FORWARD,
                                                           JOYSTICK_MAX_FORWARD));
  const float turn = fabs(reading.x) < JOYSTICK_DEADBAND
                         ? 0.0f
                         : static_cast<float>(mapDouble(reading.x, -1.0, 1.0,
                                                        -JOYSTICK_MAX_TURN,
                                                        JOYSTICK_MAX_TURN));

  Serial.print("JOY_CMD,forward,");
  Serial.print(forward);
  Serial.print(",turn,");
  Serial.println(turn);

  const float left = forward + turn;
  const float right = forward - turn;
  return DesiredWheelVel(left, right, left, right);
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
  Serial.print("ACK,");
  Serial.print(cmd.w1);
  Serial.print(",");
  Serial.print(cmd.w2);
  Serial.print(",");
  Serial.print(cmd.w3);
  Serial.print(",");
  Serial.println(cmd.w4);
}

void sendIMU() {
  // Latest fused samples from BNO08x (imu.update() is called each loop).
  AccelReadings a = imu.getAccelReadings();
  GyroReadings g = imu.getGyroReadings();

  printDebugTiming("IMU", last_imu_debug_ms);
  Serial.print("IMU,");
  Serial.print(static_cast<float>(a.ax));
  Serial.print(",");
  Serial.print(static_cast<float>(a.ay));
  Serial.print(",");
  Serial.print(static_cast<float>(a.az));
  Serial.print(",");
  Serial.print(static_cast<float>(g.rollRate));
  Serial.print(",");
  Serial.print(static_cast<float>(g.pitchRate));
  Serial.print(",");
  Serial.println(static_cast<float>(g.yawRate));
}

void setup() {
  Serial.begin(115200);
  // imu.setup();

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
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
}

void loop() {
  // imu.update();

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

    if (has_pending_cmd && (now - last_cmd_apply_ms >= CMD_APPLY_PERIOD_MS)) {
      latest_applied_cmd = latest_rx_cmd;
      applyWheelCommand(latest_applied_cmd);
      ack_dirty = true;
      has_pending_cmd = false;
      last_cmd_apply_ms = now;
    }
  } else if (now - last_joystick_apply_ms >= JOYSTICK_APPLY_PERIOD_MS) {
    if (now - last_controller_rx_ms > CONTROLLER_TIMEOUT_MS) {
      stopMotors();
      Serial.println("ERR,CONTROLLER_TIMEOUT");
    } else {
      JoystickReading wireless_reading = controllerMessage.joystick1;
      wireless_reading.x = controllerMessage.joystick2.x;
      latest_applied_cmd = joystickToWheelCommand(wireless_reading);
      applyWheelCommand(latest_applied_cmd);
      ack_dirty = true;
    }
    last_joystick_apply_ms = now;
  }

  if (ack_dirty && (now - last_ack_publish_ms >= ACK_PUBLISH_PERIOD_MS)) {
    printWheelAck(latest_applied_cmd);
    ack_dirty = false;
    last_ack_publish_ms = now;
  }

  if (now - last_imu_publish_ms >= IMU_PUBLISH_PERIOD_MS) {
    sendIMU();
    last_imu_publish_ms = now;
  }
}
