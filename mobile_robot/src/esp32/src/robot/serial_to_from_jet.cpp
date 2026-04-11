#include <Arduino.h>
#include "imu.h"
#include "robot_pinout.h"
#include "MotorDriver.h"

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
IMU imu(BNO08X_RESET, BNO08X_CS, BNO08X_INT);
GyroReadings gr;

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

bool handleWheelCommand(const String& line, DesiredWheelVel& des_wheel_spd) {
  if (!line.startsWith("WHL_CMD,")) {
    Serial.println("WRONG_START");
    return false;
  }

  float w1, w2, w3, w4;
  int parsed = sscanf(line.c_str(), "WHL_CMD,%f,%f,%f,%f", &w1, &w2, &w3, &w4);
  if (parsed != 4) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  des_wheel_spd = DesiredWheelVel(w1, w2, w3, w4);
  return true;
}

void sendIMU() {
  // Replace these with actual IMU readings
  float ax = 0.01f, ay = -0.03f, az = 9.81f;
  float gx = 0.1f, gy = 0.2f, gz = 0.3f;

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
  //imu.setup(); // uncomment for when actually connected to IMU

  for (uint8_t i = 0; i < num_wheels; i++) {
    wheels[i].setup();
  }

  elevator[0].setup();
}

void loop() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      rx_line.trim();

      DesiredWheelVel cmd;
      if (handleWheelCommand(rx_line, cmd)) {
        // Apply desired wheel commands here
        // Example:
        // wheels[0].setVelocity(cmd.w1);
        // wheels[1].setVelocity(cmd.w2);

        Serial.print("ACK,");
        Serial.print(cmd.w1); Serial.print(",");
        Serial.print(cmd.w2); Serial.print(",");
        Serial.print(cmd.w3); Serial.print(",");
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
