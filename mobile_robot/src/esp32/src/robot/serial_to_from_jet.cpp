#include <Arduino.h>
#include "imu.h"
#include "robot_pinout.h"
#include "MotorDriver.h"

// for IMU
IMU imu(BNO08X_RESET, BNO08X_CS, BNO08X_INT);
GyroReadings gr;

String rx_line = "";

#define num_wheels = 2;

// initialize two motors for wheel for now, can extend to 4 maybe
MotorDriver wheels[num_wheels] = { {A_DIR1, A_PWM1, 0}, {A_DIR2, A_PWM2, 1}}; //,
                                   //{B_DIR1, B_PWM1, 2}, {B_DIR2, B_PWM2, 3} };
MotorDriver elevator[1] = {{B_DIR1, B_PWM1, 2}}; //,
                                   //{B_DIR1, B_PWM1, 2}, {B_DIR2, B_PWM2, 3} };
                                

void setup() {
  Serial.begin(115200);
  imu.setup();

  for (uint8_t i = 0; i < NUM_MOTORS; i++)
    motors[i].setup();
}

DesiredWheelVel handleWheelCommand(const String& line) {
  if (!line.startsWith("WHL_CMD,")) return;

  float w1, w2, w3, w4;
  int parsed = sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &w1, &w2, &w3, &w4);
  if (parsed == 4) {
    DesiredWheelVel des_wheel_spd(w1, w2, w3, w4);
    return des_wheel_spd;
  }
}

void sendIMU() {
  float ax = 0.01, ay = -0.03, az = 9.81;
  float gx = 0.1, gy = 0.2, gz = 0.3;

  Serial.print("IMU,");
  Serial.print(ax); Serial.print(",");
  Serial.print(ay); Serial.print(",");
  Serial.print(az); Serial.print(",");
  Serial.print(gx); Serial.print(",");
  Serial.print(gy); Serial.print(",");
  Serial.println(gz);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      rx_line.trim();
      handleCommand(rx_line);
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  static unsigned long last_imu = 0;
  if (millis() - last_imu >= 50) {   // 20 Hz
    sendIMU();
    last_imu = millis();
  }
}