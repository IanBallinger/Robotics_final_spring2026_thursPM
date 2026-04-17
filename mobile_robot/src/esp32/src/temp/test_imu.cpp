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


// for IMU
IMU imu(13, 12, 42);
GyroReadings gr;

String rx_line = "";


void sendIMU() {
  // Replace these with actual IMU readings
  float ax = 0.01f, ay = -0.03f, az = 9.81f;
  float gx = 0.1f, gy = 0.2f, gz = 0.3f;

  printGyro(imu.getGyroReadings());

  Serial.print("IMU,");
  Serial.print(ax); Serial.print(",");
  Serial.print(ay); Serial.print(",");
  Serial.print(az); Serial.print(",");
  Serial.print(gx); Serial.print(",");
  Serial.print(gy); Serial.print(",");
  Serial.println(gz);
}

void setup() {
  Serial.begin(115200);
  imu.setup(); // uncomment for when actually connected to IMU
}

void loop() {

  static unsigned long last_imu_ms = 0;
  imu.update();

  if (millis() - last_imu_ms >= 50) {
    sendIMU();
    last_imu_ms = millis();
  }
}
