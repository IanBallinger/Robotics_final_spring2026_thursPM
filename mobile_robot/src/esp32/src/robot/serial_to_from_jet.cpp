#include <Arduino.h>
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

constexpr uint8_t num_wheels = 2;

// initialize two motors for wheel for now, can extend to 4 maybe
MotorDriver wheels[num_wheels] = {
    {A_DIR1, A_PWM1, 0},
    {A_DIR2, A_PWM2, 1}
};

MotorDriver elevator[1] = {
    {B_DIR1, B_PWM1, 2}
};

// PID
#define Kp 0.25
#define Ki 0.01
#define Kd 0
#define pidTau 0.1

EncoderVelocity encoder_lb(ENCODER1_A_PIN, ENCODER1_B_PIN, CPR_312_RPM, 0.2);
EncoderVelocity encoder_rb(ENCODER2_A_PIN, ENCODER2_B_PIN, CPR_312_RPM, 0.2);

PID pid(Kp, Ki, Kd, 0, pidTau, false);

// initialize values
double velocity_lb = 0;
double velocity_rb = 0;
double controlEffort_lb = 0;
double controlEffort_rb = 0;

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
  imu.setup();

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
  }

  elevator[0].setup();
}

void loop() {
  imu.update();

  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      rx_line.trim();

      DesiredWheelVel cmd;
      if (handleWheelCommand(rx_line, cmd)) {

        velocity_lb = encoder_lb.getVelocity(); 
        velocity_rb = encoder_rb.getVelocity(); 
        controlEffort_lb = pid.calculateParallel(velocity_lb, cmd.w1);
        controlEffort_rb = pid.calculateParallel(velocity_rb, cmd.w2);

        wheels[0].drive(controlEffort_lb);
        wheels[1].drive(controlEffort_rb);

        Serial.print("ACK,");
        Serial.print(cmd.w1);
        Serial.print(",");
        Serial.print(cmd.w2);
        Serial.print(",");
        Serial.print(cmd.w3);
        Serial.print(",");
        Serial.println(cmd.w4);
      }

      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  static unsigned long last_imu_ms = 0;
  if (millis() - last_imu_ms >= 50) {
    sendIMU();
    last_imu_ms = millis();
  }
}
