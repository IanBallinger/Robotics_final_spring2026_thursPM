#include <Arduino.h>
#include "imu.h"
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"
#include "util.h"

String rx_line = "";

constexpr uint8_t num_wheels = 2;

// ---------- timing ----------
constexpr unsigned long CONTROL_PERIOD_MS = 10;   // 100 Hz
constexpr unsigned long TELEMETRY_PERIOD_MS = 50; // 20 Hz
constexpr unsigned long CMD_TIMEOUT_MS = 250;     // stop if no command received

unsigned long last_control_ms = 0;
unsigned long last_telemetry_ms = 0;
unsigned long last_cmd_ms = 0;

// initialize two motors for wheel for now, can extend to 4 maybe
MotorDriver wheels[num_wheels] = {
    {B_DIR1, B_PWM1, 0},
    {B_DIR2, B_PWM2, 1}
};

// PID
#define Kp 0.25
#define Ki 0.01
#define Kd 0
#define pidTau 0.1

EncoderVelocity encoder_lb(ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2);
EncoderVelocity encoder_rb(ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2);

// Use one PID per wheel
PID pid_lb(Kp, Ki, Kd, 0, pidTau, false);
PID pid_rb(Kp, Ki, Kd, 0, pidTau, false);

// initialize values
double velocity_lb = 0.0;
double velocity_rb = 0.0;
double controlEffort_lb = 0.0;
double controlEffort_rb = 0.0;

// latest commanded wheel speeds
float w1 = 0.0f, w2 = 0.0f, w3 = 0.0f, w4 = 0.0f;

bool handleWheelCommand(const String& line) {
  if (!line.startsWith("WHL_CMD,")) {
    Serial.println("WRONG_START");
    return false;
  }

  int parsed = sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &w1, &w2, &w3, &w4);
  if (parsed != 4) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  last_cmd_ms = millis();

  Serial.print("ACK,");
  Serial.print(w1); Serial.print(",");
  Serial.print(w2); Serial.print(",");
  Serial.print(w3); Serial.print(",");
  Serial.println(w4);

  return true;
}

void sendEncoder() {
  Serial.print("ENC_L,");
  Serial.println(velocity_lb);

  Serial.print("ENC_R,");
  Serial.println(velocity_rb);
}

void stopMotors() {
  w1 = 0.0f;
  w2 = 0.0f;
  w3 = 0.0f;
  w4 = 0.0f;

  controlEffort_lb = 0.0;
  controlEffort_rb = 0.0;

  wheels[0].drive(0.0);
  wheels[1].drive(0.0);
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
  }

  last_control_ms = millis();
  last_telemetry_ms = millis();
  last_cmd_ms = millis();
}

void loop() {
  // -------- serial receive --------
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      rx_line.trim();
      if (rx_line.length() > 0) {
        handleWheelCommand(rx_line);
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  unsigned long now = millis();

  // -------- command timeout --------
  if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  // -------- fixed-rate control loop --------
  if (now - last_control_ms >= CONTROL_PERIOD_MS) {
    last_control_ms += CONTROL_PERIOD_MS;

    velocity_lb = encoder_lb.getVelocity();
    velocity_rb = encoder_rb.getVelocity();

    controlEffort_lb = pid_lb.calculateParallel(velocity_lb, w1);
    controlEffort_rb = pid_rb.calculateParallel(velocity_rb, w2);

    wheels[0].drive(controlEffort_lb);
    wheels[1].drive(controlEffort_rb);
  }

  // -------- fixed-rate telemetry --------
  if (now - last_telemetry_ms >= TELEMETRY_PERIOD_MS) {
    last_telemetry_ms += TELEMETRY_PERIOD_MS;
    sendEncoder();
  }
}