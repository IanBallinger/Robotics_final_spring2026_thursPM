#include <Arduino.h>
#include <ESP32Servo.h>
#include <math.h>

constexpr int ARM_SHOULDER_SERVO_PIN = 40;
constexpr int ARM_ELBOW_SERVO_PIN = 41;
constexpr int SERVO_MIN_US = 544;
constexpr int SERVO_MAX_US = 2400;
constexpr float RAD_TO_DEG_FACTOR = 180.0f / PI;

constexpr float SHOULDER_MIN_DEG = -180.0f;
constexpr float SHOULDER_MAX_DEG = 180.0f;
constexpr float ELBOW_MIN_DEG = -180.0f;
constexpr float ELBOW_MAX_DEG = 180.0f;

struct DesiredArmJointAngles {
  float theta1;
  float theta2;

  DesiredArmJointAngles() : theta1(0.0f), theta2(0.0f) {}
  DesiredArmJointAngles(float theta1_, float theta2_)
      : theta1(theta1_), theta2(theta2_) {}
};

Servo shoulder_servo;
Servo elbow_servo;
String rx_line = "";

static float clampFloat(float value, float min_value, float max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

static float convertShoulderAngleToMicroseconds(float shoulder_deg) {
  return 1360.0f + shoulder_deg * (600.0f / 90.0f);
}

static float convertElbowAngleToMicroseconds(float elbow_deg) {
  return 1090.0f - elbow_deg * (599.0f / 90.0f);
}

static bool handleArmJointAnglesCommand(
    const String& line,
    DesiredArmJointAngles& cmd) {
  if (!line.startsWith("ARM_JOINT_ANGLES_CMD,")) {
    return false;
  }

  if (sscanf(line.c_str(), "ARM_JOINT_ANGLES_CMD,%f,%f", &cmd.theta1, &cmd.theta2) != 2) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }
  return true;
}

static void moveArmToJointAngles(float theta1_rad, float theta2_rad) {
  float shoulder_deg = clampFloat(
      theta1_rad * RAD_TO_DEG_FACTOR,
      SHOULDER_MIN_DEG,
      SHOULDER_MAX_DEG);
  float elbow_deg = clampFloat(
      theta2_rad * RAD_TO_DEG_FACTOR,
      ELBOW_MIN_DEG,
      ELBOW_MAX_DEG);

  const int shoulder_us = static_cast<int>(roundf(
      convertShoulderAngleToMicroseconds(shoulder_deg)));
  const int elbow_us = static_cast<int>(roundf(
      convertElbowAngleToMicroseconds(elbow_deg)));

  shoulder_servo.writeMicroseconds(shoulder_us);
  elbow_servo.writeMicroseconds(elbow_us);

  Serial.print("ARM_JOINT_ACK,");
  Serial.print(theta1_rad, 4);
  Serial.print(",");
  Serial.print(theta2_rad, 4);
  Serial.print(",");
  Serial.print(shoulder_deg, 2);
  Serial.print(",");
  Serial.print(elbow_deg, 2);
  Serial.print(",");
  Serial.print(shoulder_us);
  Serial.print(",");
  Serial.println(elbow_us);
}

void setup() {
  Serial.begin(115200);

  shoulder_servo.setPeriodHertz(50);
  elbow_servo.setPeriodHertz(50);
  shoulder_servo.attach(ARM_SHOULDER_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  elbow_servo.attach(ARM_ELBOW_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);

  moveArmToJointAngles(0.0f, 0.0f);
  Serial.println("ARM_READY");
}

void loop() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      rx_line.trim();
      if (rx_line.length() > 0) {
        DesiredArmJointAngles cmd;
        if (handleArmJointAnglesCommand(rx_line, cmd)) {
          moveArmToJointAngles(cmd.theta1, cmd.theta2);
        } else {
          Serial.println("WRONG_START");
        }
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }
}
