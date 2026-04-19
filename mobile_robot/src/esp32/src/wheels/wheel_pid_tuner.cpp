#include <Arduino.h>
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"

struct WheelCommand {
  float w1;
  float w2;
  float w3;
  float w4;

  WheelCommand() : w1(0), w2(0), w3(0), w4(0) {}
  WheelCommand(float w1_, float w2_, float w3_, float w4_)
      : w1(w1_), w2(w2_), w3(w3_), w4(w4_) {}
};

String rx_line = "";

constexpr uint8_t NUM_WHEELS = 4;
constexpr unsigned long CONTROL_PERIOD_MS = 10;    // 100 Hz
constexpr unsigned long TELEMETRY_PERIOD_MS = 50;  // 20 Hz
constexpr unsigned long CMD_TIMEOUT_MS = 250;      // stop motors if host is silent
constexpr float PID_TAU = 0.1f;

// Adjust these signs if encoder polarity does not match commanded wheel velocity.
constexpr float ENCODER_SIGN[NUM_WHEELS] = {1.0f, -1.0f, 1.0f, -1.0f};

MotorDriver wheels[NUM_WHEELS] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3},
};

EncoderVelocity encoders[NUM_WHEELS] = {
    {ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER3_A_PIN, ENCODER3_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER4_A_PIN, ENCODER4_B_PIN, CPR_312_RPM, 0.2},
};

float kp[NUM_WHEELS] = {0.25f, 0.25f, 0.25f, 0.25f};
float ki[NUM_WHEELS] = {0.01f, 0.01f, 0.01f, 0.01f};
float kd[NUM_WHEELS] = {0.0f, 0.0f, 0.0f, 0.0f};

PID pids[NUM_WHEELS] = {
    {kp[0], ki[0], kd[0], 0.0, PID_TAU, false},
    {kp[1], ki[1], kd[1], 0.0, PID_TAU, false},
    {kp[2], ki[2], kd[2], 0.0, PID_TAU, false},
    {kp[3], ki[3], kd[3], 0.0, PID_TAU, false},
};

WheelCommand desired_cmd;
double measured_vel[NUM_WHEELS] = {0.0, 0.0, 0.0, 0.0};
double control_effort[NUM_WHEELS] = {0.0, 0.0, 0.0, 0.0};

unsigned long last_control_ms = 0;
unsigned long last_telemetry_ms = 0;
unsigned long last_cmd_ms = 0;

static void sendStatus(const char* msg) {
  Serial.print("STATUS,");
  Serial.println(msg);
}

static void sendError(const char* msg) {
  Serial.print("ERR,");
  Serial.println(msg);
}

static void sendAck() {
  Serial.print("ACK,");
  Serial.print(desired_cmd.w1);
  Serial.print(",");
  Serial.print(desired_cmd.w2);
  Serial.print(",");
  Serial.print(desired_cmd.w3);
  Serial.print(",");
  Serial.println(desired_cmd.w4);
}

static void sendEncoderTelemetry() {
  Serial.print("ENC,");
  Serial.print(measured_vel[0]);
  Serial.print(",");
  Serial.print(measured_vel[1]);
  Serial.print(",");
  Serial.print(measured_vel[2]);
  Serial.print(",");
  Serial.println(measured_vel[3]);
}

static void sendEffortTelemetry() {
  Serial.print("EFF,");
  Serial.print(control_effort[0]);
  Serial.print(",");
  Serial.print(control_effort[1]);
  Serial.print(",");
  Serial.print(control_effort[2]);
  Serial.print(",");
  Serial.println(control_effort[3]);
}

static void sendPidTelemetry() {
  Serial.print("PID,");
  Serial.print(kp[0]);
  Serial.print(",");
  Serial.print(ki[0]);
  Serial.print(",");
  Serial.print(kd[0]);
  Serial.print(",");
  Serial.print(kp[1]);
  Serial.print(",");
  Serial.print(ki[1]);
  Serial.print(",");
  Serial.print(kd[1]);
  Serial.print(",");
  Serial.print(kp[2]);
  Serial.print(",");
  Serial.print(ki[2]);
  Serial.print(",");
  Serial.print(kd[2]);
  Serial.print(",");
  Serial.print(kp[3]);
  Serial.print(",");
  Serial.print(ki[3]);
  Serial.print(",");
  Serial.println(kd[3]);
}

static void stopMotors() {
  desired_cmd = WheelCommand();
  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    control_effort[i] = 0.0;
    wheels[i].drive(0.0);
  }
}

static void updatePidWheel(uint8_t wheel_idx, float new_kp, float new_ki,
                           float new_kd) {
  kp[wheel_idx] = new_kp;
  ki[wheel_idx] = new_ki;
  kd[wheel_idx] = new_kd;
  pids[wheel_idx].setParallelTunings(new_kp, new_ki, new_kd);
}

static bool handleWheelCommand(const String& line) {
  float w1 = 0.0f, w2 = 0.0f, w3 = 0.0f, w4 = 0.0f;
  if (sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &w1, &w2, &w3, &w4) != 4) {
    return false;
  }

  desired_cmd = WheelCommand(w1, w2, w3, w4);
  last_cmd_ms = millis();
  sendAck();
  return true;
}

static bool handlePidAllCommand(const String& line) {
  float new_kp = 0.0f, new_ki = 0.0f, new_kd = 0.0f;
  if (sscanf(line.c_str(), "PID_ALL,%f,%f,%f", &new_kp, &new_ki, &new_kd) != 3) {
    return false;
  }

  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    updatePidWheel(i, new_kp, new_ki, new_kd);
  }
  sendPidTelemetry();
  return true;
}

static bool handlePidWheelCommand(const String& line) {
  int wheel = 0;
  float new_kp = 0.0f, new_ki = 0.0f, new_kd = 0.0f;
  if (sscanf(line.c_str(), "PID_WHEEL,%d,%f,%f,%f", &wheel, &new_kp, &new_ki,
             &new_kd) != 4) {
    return false;
  }

  if (wheel < 1 || wheel > 4) {
    sendError("BAD_WHEEL_INDEX");
    return true;
  }

  updatePidWheel(static_cast<uint8_t>(wheel - 1), new_kp, new_ki, new_kd);
  sendPidTelemetry();
  return true;
}

static void handleCommand(const String& line) {
  if (line.startsWith("WHL_CMD,")) {
    if (!handleWheelCommand(line)) {
      sendError("BAD_WHL_CMD");
    }
    return;
  }

  if (line.startsWith("PID_ALL,")) {
    if (!handlePidAllCommand(line)) {
      sendError("BAD_PID_ALL");
    }
    return;
  }

  if (line.startsWith("PID_WHEEL,")) {
    if (!handlePidWheelCommand(line)) {
      sendError("BAD_PID_WHEEL");
    }
    return;
  }

  if (line == "ZERO") {
    stopMotors();
    sendAck();
    return;
  }

  if (line == "GET_PID") {
    sendPidTelemetry();
    return;
  }

  sendError("UNKNOWN_CMD");
}

static void updateControl() {
  const float setpoints[NUM_WHEELS] = {
      desired_cmd.w1, desired_cmd.w2, desired_cmd.w3, desired_cmd.w4,
  };

  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    measured_vel[i] = ENCODER_SIGN[i] * encoders[i].getVelocity();
    control_effort[i] = pids[i].calculateParallel(measured_vel[i], setpoints[i]);
    wheels[i].drive(control_effort[i]);
  }
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    wheels[i].setup();
  }

  const unsigned long now = millis();
  last_control_ms = now;
  last_telemetry_ms = now;
  last_cmd_ms = now;

  sendStatus("WHEEL_PID_TUNER_READY");
  sendPidTelemetry();
}

void loop() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      rx_line.trim();
      if (rx_line.length() > 0) {
        handleCommand(rx_line);
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();

  if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  if (now - last_control_ms >= CONTROL_PERIOD_MS) {
    last_control_ms += CONTROL_PERIOD_MS;
    updateControl();
  }

  if (now - last_telemetry_ms >= TELEMETRY_PERIOD_MS) {
    last_telemetry_ms += TELEMETRY_PERIOD_MS;
    sendEncoderTelemetry();
    sendEffortTelemetry();
  }
}
