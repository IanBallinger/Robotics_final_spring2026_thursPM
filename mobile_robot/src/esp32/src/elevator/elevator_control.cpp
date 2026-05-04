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

struct DesiredArmJointAngles {
  float theta1;
  float theta2;

  DesiredArmJointAngles() : theta1(0.0f), theta2(0.0f) {}
  DesiredArmJointAngles(float theta1_, float theta2_) : theta1(theta1_), theta2(theta2_) {}
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
constexpr float ELEVATOR_ENCODER_TAU_S = 0.5f;
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
constexpr unsigned long ARM_UPDATE_PERIOD_MS = 20;
constexpr float ARM_SHOULDER_RATE_RAD_S = 1.2f;
constexpr float ARM_ELBOW_RATE_RAD_S = 1.2f;
constexpr float ARM_JOINT_REACHED_EPS_RAD = 0.01f;

DesiredElevatorState latest_rx_cmd;
DesiredElevatorState latest_applied_cmd;
DesiredArmPosition latest_arm_cmd;
DesiredArmJointAngles latest_arm_joint_angles_cmd;
bool has_valid_elevator_cmd = false;
bool ack_dirty = false;
unsigned long last_cmd_rx_ms = 0;
unsigned long last_cmd_apply_ms = 0;
unsigned long last_meas_publish_ms = 0;
unsigned long last_ack_publish_ms = 0;
unsigned long last_ack_debug_ms = 0;
unsigned long last_meas_debug_ms = 0;
unsigned long last_arm_update_ms = 0;

bool has_valid_arm_target = false;
float target_shoulder_rad = 0.0f;
float target_elbow_rad = 0.0f;
float applied_shoulder_rad = 0.0f;
float applied_elbow_rad = 0.0f;

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

constexpr float ELEVATOR_ENCODER_MAX_SPEED_MPS = 1.0f;
constexpr float ELEVATOR_ENCODER_JUMP_MARGIN_M = 0.01f;
constexpr float ELEVATOR_TOF_CORRECTION_GAIN = 0.15f;
constexpr float ELEVATOR_ENCODER_REANCHOR_GAIN = 0.05f;
constexpr float ELEVATOR_ENCODER_REANCHOR_WINDOW_M = 0.03f;

bool elevator_height_filter_initialized = false;
bool last_encoder_height_valid = false;
float fused_elevator_height_m = NAN;
float last_encoder_height_m = NAN;
float last_tof_height_m = NAN;
unsigned long last_height_update_ms = 0;

#define Kp 3.5f
#define Ki 0.1f
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

static bool handleArmJointAnglesCommand(const String& line, DesiredArmJointAngles& cmd) {
  if (!line.startsWith("ARM_JOINT_ANGLES_CMD,")) {
    return false;
  }

  if (sscanf(line.c_str(), "ARM_JOINT_ANGLES_CMD,%f,%f", &cmd.theta1, &cmd.theta2) != 2) {
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
  const unsigned long now_ms = millis();
  const float tof_height_m = readTofElevatorHeightMeters();
  last_tof_height_m = tof_height_m;
  maybeInitializeEncoderHeight(tof_height_m);

  const float encoder_height_m = readEncoderHeightMeters();

  if (!elevator_height_filter_initialized) {
    if (!isnan(tof_height_m)) {
      fused_elevator_height_m = tof_height_m;
      if (!isnan(encoder_height_m)) {
        last_encoder_height_m = encoder_height_m;
        last_encoder_height_valid = true;
      }
      elevator_height_filter_initialized = true;
    } else if (!isnan(encoder_height_m)) {
      fused_elevator_height_m = encoder_height_m;
      last_encoder_height_m = encoder_height_m;
      last_encoder_height_valid = true;
      elevator_height_filter_initialized = true;
    }
    last_height_update_ms = now_ms;
    return fused_elevator_height_m;
  }

  const float dt_s = (last_height_update_ms == 0 || now_ms <= last_height_update_ms)
                         ? 0.0f
                         : (now_ms - last_height_update_ms) * 0.001f;
  last_height_update_ms = now_ms;

  float estimated_height_m = fused_elevator_height_m;

  if (!isnan(encoder_height_m)) {
    if (last_encoder_height_valid && dt_s > 0.0f) {
      const float encoder_delta_m = encoder_height_m - last_encoder_height_m;
      const float max_allowed_delta_m =
          ELEVATOR_ENCODER_MAX_SPEED_MPS * dt_s + ELEVATOR_ENCODER_JUMP_MARGIN_M;
      if (fabsf(encoder_delta_m) <= max_allowed_delta_m) {
        estimated_height_m = fused_elevator_height_m + encoder_delta_m;
      }
    } else {
      estimated_height_m = encoder_height_m;
    }

    last_encoder_height_m = encoder_height_m;
    last_encoder_height_valid = true;
  } else {
    last_encoder_height_valid = false;
  }

  if (!isnan(tof_height_m)) {
    if (isnan(estimated_height_m)) {
      estimated_height_m = tof_height_m;
    } else {
      estimated_height_m +=
          ELEVATOR_TOF_CORRECTION_GAIN * (tof_height_m - estimated_height_m);
    }

    if (!isnan(encoder_height_m) &&
        fabsf(tof_height_m - encoder_height_m) <= ELEVATOR_ENCODER_REANCHOR_WINDOW_M) {
      encoder_zero_height_m +=
          ELEVATOR_ENCODER_REANCHOR_GAIN * (tof_height_m - encoder_height_m);
      last_encoder_height_m = readEncoderHeightMeters();
    }
  }

  fused_elevator_height_m = estimated_height_m;
  return fused_elevator_height_m;
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

static bool commandArmToJointAngles(const DesiredArmJointAngles& cmd) {
  target_shoulder_rad = cmd.theta1;
  target_elbow_rad = cmd.theta2;
  has_valid_arm_target = true;

  Serial.print("ARM_JOINT_ACK,");
  Serial.print(cmd.theta1, 4);
  Serial.print(",");
  Serial.print(cmd.theta2, 4);
  Serial.print(",");
  Serial.print(target_shoulder_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(target_elbow_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(applied_shoulder_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.println(applied_elbow_rad * RAD_TO_DEG_FACTOR, 2);
  return true;
}

static float stepToward(float current_value, float target_value, float max_step) {
  const float error = target_value - current_value;
  if (fabsf(error) <= max_step) {
    return target_value;
  }
  return current_value + copysignf(max_step, error);
}

static void updateArmMotion(unsigned long now_ms) {
  if (!has_valid_arm_target) {
    return;
  }
  if (last_arm_update_ms != 0 && now_ms - last_arm_update_ms < ARM_UPDATE_PERIOD_MS) {
    return;
  }

  const float dt_s = (last_arm_update_ms == 0)
                         ? (ARM_UPDATE_PERIOD_MS * 0.001f)
                         : ((now_ms - last_arm_update_ms) * 0.001f);
  last_arm_update_ms = now_ms;

  applied_shoulder_rad = stepToward(
      applied_shoulder_rad,
      target_shoulder_rad,
      ARM_SHOULDER_RATE_RAD_S * dt_s);
  applied_elbow_rad = stepToward(
      applied_elbow_rad,
      target_elbow_rad,
      ARM_ELBOW_RATE_RAD_S * dt_s);

  moveArmToJointAngles(applied_shoulder_rad, applied_elbow_rad);

  if (fabsf(target_shoulder_rad - applied_shoulder_rad) <= ARM_JOINT_REACHED_EPS_RAD &&
      fabsf(target_elbow_rad - applied_elbow_rad) <= ARM_JOINT_REACHED_EPS_RAD) {
    applied_shoulder_rad = target_shoulder_rad;
    applied_elbow_rad = target_elbow_rad;
    moveArmToJointAngles(applied_shoulder_rad, applied_elbow_rad);
  }
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

  target_shoulder_rad = joint_angles.first;
  target_elbow_rad = joint_angles.second;
  has_valid_arm_target = true;

  Serial.print("ARM_ACK,");
  Serial.print(cmd.xE, 4);
  Serial.print(",");
  Serial.print(cmd.yE, 4);
  Serial.print(",");
  Serial.print(target_shoulder_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(target_elbow_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.print(applied_shoulder_rad * RAD_TO_DEG_FACTOR, 2);
  Serial.print(",");
  Serial.println(applied_elbow_rad * RAD_TO_DEG_FACTOR, 2);
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
  applied_shoulder_rad = 0.0f;
  applied_elbow_rad = 0.0f;
  target_shoulder_rad = 0.0f;
  target_elbow_rad = 0.0f;

  const unsigned long now = millis();
  last_cmd_rx_ms = now;
  last_cmd_apply_ms = now;
  last_meas_publish_ms = now;
  last_ack_publish_ms = now;
  last_arm_update_ms = now;

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
          DesiredArmJointAngles arm_joint_angles_cmd;
          if (handleArmJointAnglesCommand(rx_line, arm_joint_angles_cmd)) {
            latest_arm_joint_angles_cmd = arm_joint_angles_cmd;
            commandArmToJointAngles(latest_arm_joint_angles_cmd);
          } else {
            Serial.println("WRONG_START");
          }
        }
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();
  updateArmMotion(now);
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
          Serial.println("WRONG_START");
        }
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();
  updateArmMotion(now);
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
