#include <Arduino.h>
#include <cstddef>
#include "imu.h"
#include "robot_pinout.h"
#include "MotorDriver.h"
#include "EncoderVelocity.h"
#include "PID.h"
#include "util.h"

// If this type already exists elsewhere in your project, remove this struct.
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

// FIFO of incoming wheel commands; drained on each IMU print interval.
static constexpr size_t kWheelCmdQCap = 32;
static DesiredWheelVel wheel_cmd_q[kWheelCmdQCap];
static size_t wheel_q_head = 0;
static size_t wheel_q_count = 0;

static bool wheel_cmd_enqueue(const DesiredWheelVel& v) {
  if (wheel_q_count >= kWheelCmdQCap) {
    return false;
  }
  const size_t idx = (wheel_q_head + wheel_q_count) % kWheelCmdQCap;
  wheel_cmd_q[idx] = v;
  wheel_q_count++;
  return true;
}

static bool wheel_cmd_dequeue(DesiredWheelVel& out) {
  if (wheel_q_count == 0) {
    return false;
  }
  out = wheel_cmd_q[wheel_q_head];
  wheel_q_head = (wheel_q_head + 1) % kWheelCmdQCap;
  wheel_q_count--;
  return true;
}

constexpr uint8_t num_wheels = 4;

// initialize two motors for wheel for now, can extend to 4 maybe
MotorDriver wheels[num_wheels] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1},
    {B_DIR1, B_PWM1, 2},
    {B_DIR2, B_PWM2, 3}
};

// TODO: define pinout for elevator (might be on a different motor driver and serial connection)
// MotorDriver elevator[1] = {
//     {B_DIR1, B_PWM1, 2}
// };

//     FRONT
// [1]-------[2]
//  |         |
//  |  ROBOT  |
//  |         |
// [3]-------[4]
//     REAR


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

// initialize values
double velocity1 = 0;
double velocity2 = 0;
double velocity3 = 0;
double velocity4 = 0;
double controlEffort1 = 0;
double controlEffort2 = 0;
double controlEffort3 = 0;
double controlEffort4 = 0;

//TODO: handle if we get joystick data to take over manual control

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

static void printWheelAck(const DesiredWheelVel& cmd) {
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

}

void loop() {
  // imu.update();

  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      rx_line.trim();

      DesiredWheelVel cmd;
      if (handleWheelCommand(rx_line, cmd)) {
        if (!wheel_cmd_enqueue(cmd)) {
          Serial.println("WHL_Q_OVERFLOW");
        }
      }

      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  static unsigned long last_imu_ms = 0;
  if (millis() - last_imu_ms >= 50) {
    sendIMU();
    DesiredWheelVel cmd;
    while (wheel_cmd_dequeue(cmd)) {
      applyWheelCommand(cmd);
      printWheelAck(cmd);
    }
    last_imu_ms = millis();
  }
}
