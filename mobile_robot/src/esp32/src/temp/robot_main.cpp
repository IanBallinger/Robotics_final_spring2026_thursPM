#include <Arduino.h>
#include "robot_drive.h"
#include "wireless.h"
#include "util.h"
#include "robot_motion_control.h"
#include "imu.h"

extern IMU imu;
extern GyroReadings gr;


void setup() {
    Serial.begin(115200);
    setupDrive();
    imu.setup();
    setupWireless();
}

void loop() {
    // Update velocity setpoints based on trajectory at 50Hz
    EVERY_N_MILLIS(20) {
        followTrajectory();
    }

    // Update PID at 200Hz
    EVERY_N_MILLIS(5) {
        updatePIDs();
    }

    // Send and print robot values at 20Hz
    EVERY_N_MILLIS(50) {
        imu.update();
        updateOdometry();
        sendRobotData();

        Serial.printf("x: %.2f, y: %.2f, theta: %.2f,  r: %.2f\n",
                    robotMessage.x, robotMessage.y, robotMessage.theta, gr.yawRate);
    }
  
}