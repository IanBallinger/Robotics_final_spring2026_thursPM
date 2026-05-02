#include "Adafruit_VL53L0X.h"
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <ESP32Servo.h>
#include "PID.h"
#include <esp_now.h>
#include <WiFi.h>
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "PID.h"
#include "util.h"
#include "arm_control.h"
#include "EncoderVelocity.h"


#define TCAADDR 0x70

struct DesiredElevatorState {
  float height_m;

  DesiredElevatorState() : height_m(0.0f) {}
  explicit DesiredElevatorState(float height_m_) : height_m(height_m_) {}
};

// Desired arms position measured from the base joint
// TODO: will need to adjust for the position of the camera, will come from the Jetson
struct DesiredArmPosition {
  float xE;
  float yE;

  DesiredArmPosition() : xE(0.0f), yE(0.0f) {}
  DesiredArmPosition(float xE_, float yE_) : xE(xE_), yE(yE_) {}
};

Adafruit_VL53L0X lox = Adafruit_VL53L0X();
String rx_line = "";

constexpr uint8_t TOF_MUX_CHANNEL = 0;
constexpr unsigned long CMD_APPLY_PERIOD_MS = 50;
constexpr unsigned long MEAS_PUBLISH_PERIOD_MS = 50;
constexpr unsigned long ACK_PUBLISH_PERIOD_MS = 50;
constexpr unsigned long CMD_TIMEOUT_MS = 250;
constexpr bool SERIAL_DEBUG_TIMING = true;
constexpr float MM_TO_M = 1.0f / 1000.0f;

// Elevator encoder configuration.
// By default this uses ENCODER1 on the elevator ESP32. Tune
// ELEVATOR_METERS_PER_RAD and ELEVATOR_ENCODER_SIGN to match the pulley/lead
// screw geometry and wiring direction.
constexpr int ELEVATOR_ENCODER_A_PIN = 4;
constexpr int ELEVATOR_ENCODER_B_PIN = 5;
constexpr int ELEVATOR_ENCODER_CPR = CPR_312_RPM;
constexpr float ELEVATOR_ENCODER_TAU_S = 0.05f;
constexpr float ELEVATOR_ENCODER_SIGN = 1.0f;
constexpr float ELEVATOR_METERS_PER_RAD = 0.018f;
constexpr float ELEVATOR_METERS_BIAS = -0.02f;

// Servo Arm setup and constants
constexpr int ARM_SHOULDER_SERVO_PIN = 40;
constexpr int ARM_ELBOW_SERVO_PIN = 41;
constexpr int SERVO_MIN_US = 544;
constexpr int SERVO_MAX_US = 2400;
constexpr float RAD_TO_DEG_FACTOR = 180.0f / PI;
constexpr float ARM_BASE_X_M = 0.0f;
constexpr float ARM_BASE_Y_M = 0.0f;
constexpr float ARM_LINK_1_M = 0.26f;
constexpr float ARM_LINK_2_M = 0.16f; // TODO: need to add length of end effector

constexpr float SHOULDER_MIN_DEG = -180.0f; //changed from 0-90 deg
constexpr float SHOULDER_MAX_DEG = 180.0f;
constexpr float ELBOW_MIN_DEG = -180.0f;
constexpr float ELBOW_MAX_DEG = 180.0f;

DesiredElevatorState latest_rx_cmd;
DesiredElevatorState latest_applied_cmd;
DesiredArmPosition latest_arm_cmd;
bool has_valid_elevator_cmd = false;
bool ack_dirty = false;
unsigned long last_cmd_rx_ms = 0;
unsigned long last_cmd_apply_ms = 0;
unsigned long last_meas_publish_ms = 0;
unsigned long last_ack_publish_ms = 0;
unsigned long last_ack_debug_ms = 0;
unsigned long last_meas_debug_ms = 0;

// TODO: need to update these pins
MotorDriver elevator_driver {B_DIR1, B_PWM1, 0};
EncoderVelocity elevator_encoder {ELEVATOR_ENCODER_A_PIN,
                                  ELEVATOR_ENCODER_B_PIN,
                                  ELEVATOR_ENCODER_CPR,
                                  ELEVATOR_ENCODER_TAU_S};
Servo shoulder_servo;
Servo elbow_servo;

bool encoder_height_initialized = false;
float encoder_zero_height_m = 0.0f;
float encoder_zero_position_rad = 0.0f;

float last_tof_height_m = NAN;

#define Kp 3.0f
#define Ki 0.0f
#define Kd 0.0f

double integral_min = -1e6;
double integral_max = 1e6;

PID pid = {Kp, Ki, Kd, 0.0, 0.1f, false};

// For debugging
double shoulder_ms = 0.0;
double elbow_ms = 0.0;

void tcaSelect(uint8_t channel) {
  if (channel > 7) {
    return;
  }
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
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

static float clampFloat(float value, float min_value, float max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

static float convertShoulderAngleToMicroseconds (const float shoulder_angle){
  // 0 deg: 1360
  // 90 deg: 1960

  // shoulder angle should be between 0 and 90
  return 1360 + shoulder_angle * (600.0/90.0);
}

static float convertElbowAngleToMicroseconds (const float elbow_angle){
  // 0 deg: 1090
  // 90 deg: 1691
  // 180 deg: 2287

  // elbow angle should be between 0 and 180
  // have to negate desired elbow angle to actual command
  return 1090 - elbow_angle * (599.0/90.0);
}

static bool handleElevatorCommand(const String& line, DesiredElevatorState& cmd) {
  if (!line.startsWith("ELV_CMD,")) {
    return false;
  }

  if (sscanf(line.c_str(), "ELV_CMD,%f", &cmd.height_m) != 1) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  return true;
}

static bool handleArmCommand(const String& line, DesiredArmPosition& cmd) {
  if (!line.startsWith("ARM_CMD,")) {
    return false;
  }

  if (sscanf(line.c_str(), "ARM_CMD,%f,%f", &cmd.xE, &cmd.yE) != 2) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  return true;
}

static float readTofElevatorHeightMeters() {
  tcaSelect(TOF_MUX_CHANNEL);
  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus == 4) {
    return NAN;
  }

  return static_cast<float>(measure.RangeMilliMeter) * MM_TO_M;
}

static void maybeInitializeEncoderHeight(float tof_height_m) {
  if (encoder_height_initialized || isnan(tof_height_m)) {
    return;
  }
  encoder_zero_height_m = tof_height_m;
  encoder_zero_position_rad = ELEVATOR_ENCODER_SIGN * elevator_encoder.getPosition();
  encoder_height_initialized = true;
}

static float readEncoderHeightMeters() {
  if (!encoder_height_initialized) {
    return NAN;
  }
  const float position_rad = ELEVATOR_ENCODER_SIGN * elevator_encoder.getPosition();
  return encoder_zero_height_m + ELEVATOR_METERS_BIAS +
         (position_rad - encoder_zero_position_rad) * ELEVATOR_METERS_PER_RAD;
}

static float readElevatorHeightMeters() {
  const float tof_height_m = readTofElevatorHeightMeters();
  last_tof_height_m = tof_height_m;

  // Use the ToF sensor as the primary measurement, matching the behavior of
  // the known-good implementation from commit 97aab57. The encoder is only
  // used as a fallback when the ToF sample is temporarily invalid.
  if (!isnan(tof_height_m)) {
    maybeInitializeEncoderHeight(tof_height_m);
    return tof_height_m;
  }

  return readEncoderHeightMeters();
}

static void applyElevatorCommand(const DesiredElevatorState& cmd,
                                 float measured_height_m) {
  latest_applied_cmd = cmd;

  double control_effort = pid.calculateParallel(measured_height_m, cmd.height_m);
  elevator_driver.drive(control_effort);
  Serial.print("ELV_CTRL_EFF,");
  Serial.println(control_effort, 4);
}

static void printElevatorAck(const DesiredElevatorState& cmd) {
  printDebugTiming("ELV_ACK", last_ack_debug_ms);
  Serial.print("ELV_ACK,");
  Serial.println(cmd.height_m, 4);
}

static bool moveArmToJointAngles(float theta1_rad, float theta2_rad) {
  float shoulder_deg = theta1_rad * RAD_TO_DEG_FACTOR;
  float elbow_deg = theta2_rad * RAD_TO_DEG_FACTOR;

  shoulder_deg = clampFloat(shoulder_deg, SHOULDER_MIN_DEG, SHOULDER_MAX_DEG);
  elbow_deg = clampFloat(elbow_deg, ELBOW_MIN_DEG, ELBOW_MAX_DEG);

  shoulder_ms = convertShoulderAngleToMicroseconds(shoulder_deg);
  elbow_ms = convertElbowAngleToMicroseconds(elbow_deg);

  shoulder_servo.writeMicroseconds(shoulder_ms);
  elbow_servo.writeMicroseconds(elbow_ms);
  return true;
}

static bool moveArmToXY(const DesiredArmPosition& cmd) {
  const auto joint_angles = inverseKinematics(
      cmd.xE,
      cmd.yE,
      ARM_BASE_X_M,
      ARM_BASE_Y_M,
      ARM_LINK_1_M,
      ARM_LINK_2_M);

  if (isnan(joint_angles.first) || isnan(joint_angles.second)) {
    Serial.println("ARM_UNREACHABLE");
    return false;
  }

  // Hard-coding for debugging, change to joint_angles.first, joint_angles.second
  moveArmToJointAngles(joint_angles.first, joint_angles.second);
  Serial.print("ARM_ACK,");
  Serial.print(cmd.xE, 4);
  Serial.print(",");
  Serial.print(cmd.yE, 4);
  Serial.print(",");
  Serial.print(joint_angles.first * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(joint_angles.second * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(shoulder_ms);
  Serial.print(",");
  Serial.println(elbow_ms);
  return true;
}

static void publishElevatorMeasurement(float height_m) {
  printDebugTiming("ELV_MEAS", last_meas_debug_ms);
  Serial.print("ELV_MEAS,");
  Serial.println(height_m, 4);

  const float encoder_position_rad = ELEVATOR_ENCODER_SIGN * elevator_encoder.getPosition();
  const float encoder_velocity_rad_s = ELEVATOR_ENCODER_SIGN * elevator_encoder.getVelocity();
  Serial.print("ELV_ENC,");
  Serial.print(encoder_position_rad, 4);
  Serial.print(",");
  Serial.print(encoder_velocity_rad_s, 4);
  Serial.print(",");
  if (encoder_height_initialized) {
    Serial.print(readEncoderHeightMeters(), 4);
  } else {
    Serial.print("nan");
  }
  Serial.print(",");
  if (!isnan(last_tof_height_m)) {
    Serial.println(last_tof_height_m, 4);
  } else {
    Serial.println("nan");
  }
}

static void stopElevator() {
  latest_rx_cmd = DesiredElevatorState();
  latest_applied_cmd = DesiredElevatorState();
  has_valid_elevator_cmd = false;
  ack_dirty = false;

  elevator_driver.drive(0.0);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  tcaSelect(TOF_MUX_CHANNEL);
  if (!lox.begin()) {
    Serial.println("ERR,VL53L0X_INIT_FAILED");
    // TODO: fix I2C connection, this is just for debugging
    // while (1) {
    //   delay(100);
    // }
  }

  elevator_driver.setup();
  pid.setParallelTunings(Kp, Ki, Kd, 0.1f, integral_min, integral_max);
  Serial.println("SETUP_PID");

  shoulder_servo.setPeriodHertz(50);
  Serial.println("SETUP_SHLDR");
  elbow_servo.setPeriodHertz(50);
  Serial.println("SETUP_ELBOW");
  shoulder_servo.attach(ARM_SHOULDER_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  elbow_servo.attach(ARM_ELBOW_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  moveArmToJointAngles(0.0f, 0.0f);

  const unsigned long now = millis();
  last_cmd_rx_ms = now;
  last_cmd_apply_ms = now;
  last_meas_publish_ms = now;
  last_ack_publish_ms = now;

  Serial.println("ELV_READY");
}

void loop() {
  
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      rx_line.trim();
      DesiredElevatorState cmd;
      DesiredArmPosition arm_cmd;
      if (rx_line.length() > 0) {
        if (handleElevatorCommand(rx_line, cmd)) {
          latest_rx_cmd = cmd;
          has_valid_elevator_cmd = true;
          last_cmd_rx_ms = millis();
        } else if (handleArmCommand(rx_line, arm_cmd)) {
          latest_arm_cmd = arm_cmd;
          moveArmToXY(latest_arm_cmd);
        } else {
          Serial.println("WRONG_START");
        }
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();
  const float measured_height_m = readElevatorHeightMeters();

  if (now - last_cmd_rx_ms > CMD_TIMEOUT_MS) {
    stopElevator();
  }

  if (has_valid_elevator_cmd && now - last_cmd_apply_ms >= CMD_APPLY_PERIOD_MS) {
    if (!isnan(measured_height_m)) {
      applyElevatorCommand(latest_rx_cmd, measured_height_m);
    } else {
      elevator_driver.drive(0.0);
    }
    ack_dirty = true;
    last_cmd_apply_ms = now;
  }

  if (ack_dirty && now - last_ack_publish_ms >= ACK_PUBLISH_PERIOD_MS) {
     printElevatorAck(latest_applied_cmd);
     ack_dirty = false;
     last_ack_publish_ms = now;
  }

  if (!isnan(measured_height_m) && now - last_meas_publish_ms >= MEAS_PUBLISH_PERIOD_MS) {
    publishElevatorMeasurement(measured_height_m);
    last_meas_publish_ms = now;
  }
}
