#include <Arduino.h>
#include <cstdint>
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"
#include "util.h"

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
constexpr unsigned long CONTROL_PERIOD_MS = 5;     // 20 Hz, matching motor_velocity_control.cpp
constexpr unsigned long TELEMETRY_PERIOD_MS = 50;  // 20 Hz
constexpr unsigned long CMD_TIMEOUT_MS = 250;      // stop motors if host is silent
constexpr float PID_TAU = 0.1f;

// Match the alternating encoder polarity used by the working multi-wheel drive code.
constexpr float ENCODER_SIGN[NUM_WHEELS] = {1.0f, -1.0f, 1.0f, -1.0f};

MotorDriver wheels[NUM_WHEELS] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3},
};

EncoderVelocity encoders[NUM_WHEELS] = {
    {ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER3_A_PIN, ENCODER3_B_PIN, CPR_312_RPM, 0.2},
    {ENCODER4_A_PIN, ENCODER4_B_PIN, CPR_312_RPM, 0.2},
};

float kp[NUM_WHEELS] = {0.4f, 0.38f, 0.4f, 0.4f};
float ki[NUM_WHEELS] = {0.5f, 0.7f, 0.7f, 0.5f};
float kd[NUM_WHEELS] = {0.0f, 0.0f, 0.0f, 0.0f}; // haven't tuned this yet

PID pid1 = {kp[0], ki[0], kd[0], 0.0, PID_TAU, false};
PID pid2 = {kp[1], ki[1], kd[1], 0.0, PID_TAU, false};
PID pid3 = {kp[2], ki[2], kd[2], 0.0, PID_TAU, false};
PID pid4 = {kp[3], ki[3], kd[3], 0.0, PID_TAU, false};
PID pids[NUM_WHEELS] = {pid1, pid2, pid3, pid4};

// Keep integral anti-windup symmetric so negative setpoints can build
// the same integral action as positive setpoints.
double integral_min = -1e6;
double integral_max = 1e6;

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

static void stopMotors() {
  desired_cmd = WheelCommand();
  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    control_effort[i] = 0.0;
    wheels[i].drive(0.0);
  }
}

static bool handleWheelCommand(const String& line) {
  float w1 = 0.0f, w2 = 0.0f, w3 = 0.0f, w4 = 0.0f;
  if (sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &w1, &w2, &w3, &w4) != 4) {
    return false;
  }

  // switch directions for w2 and w3
  w2 = w2 * (-1.0);
  w3 = w3 * (-1.0);

  desired_cmd = WheelCommand(w1, w2, w3, w4);
  last_cmd_ms = millis();
  return true;
}

static void handleCommand(const String& line) {
  if (line.startsWith("WHL_CMD,")) {
    if (!handleWheelCommand(line)) {
      sendError("BAD_WHL_CMD");
    }
    return;
  }
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

  // set the PID tunings
  for (uint8_t i = 0; i < NUM_WHEELS; ++i) {
    pids[i].setParallelTunings(kp[i], ki[i], kd[i], PID_TAU, integral_min, integral_max);
  }

  sendStatus("WHEEL_PID_TUNER_READY");
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
    sendError("CMD_TIMEOUT");
  }

  if (now - last_control_ms >= CONTROL_PERIOD_MS) {
    last_control_ms += CONTROL_PERIOD_MS;
    updateControl();
  }

  if (now - last_telemetry_ms >= TELEMETRY_PERIOD_MS) {
    last_telemetry_ms += TELEMETRY_PERIOD_MS;
    sendEncoderTelemetry();
    // sendEffortTelemetry();
  }
}
