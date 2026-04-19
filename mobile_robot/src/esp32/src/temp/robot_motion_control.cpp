#include <Arduino.h>
#include "util.h"
#include "robot_drive.h"
#include "EncoderVelocity.h"
#include "wireless.h"
#include "robot_motion_control.h"
#include "robot_pinout.h"

// #define UTURN
// #define CIRCLE
// #define JOYSTICK
#define YOUR_TRAJECTORY

extern RobotMessage robotMessage;
extern ControllerMessage controllerMessage;

int state = 0;
double robotVelocity = 0; // velocity of robot, in m/s
double k = 0; // curvature k is 1/radius from center of rotation circle

extern EncoderVelocity encoders[NUM_MOTORS];
double currPhiL = 0;
double currPhiR = 0;
double prevPhiL = 0;
double prevPhiR = 0;

// for IMU
IMU imu(BNO08X_RESET, BNO08X_CS, BNO08X_INT);
GyroReadings gr;

// Sets the desired wheel velocities based on desired robot velocity in m/s
// and k curvature in 1/m representing 1/(radius of curvature)
void setWheelVelocities(float robotVelocity, float k){
    double left = (robotVelocity - k * b_base * robotVelocity) / r_wheel;
    double right = 2 * robotVelocity / r_wheel  - left;
    updateSetpoints(left, right);
}

// Makes robot follow a trajectory
void followTrajectory() {

    #ifdef JOYSTICK
    if (freshWirelessData) {
        double forward = abs(controllerMessage.joystick1.y) < 0.1 ? 0 : mapDouble(controllerMessage.joystick1.y, -1, 1, -MAX_FORWARD, MAX_FORWARD);
        double turn = abs(controllerMessage.joystick1.x) < 0.1 ? 0 : mapDouble(controllerMessage.joystick1.x, -1, 1, -MAX_TURN, MAX_TURN);
        updateSetpoints(forward + turn, forward - turn);
    }
    #endif 

    #ifdef CIRCLE
    robotVelocity = 0.2;
    k = 1/0.5;
    setWheelVelocities(robotVelocity, k);
    #endif 

    #ifdef UTURN
    switch (state) {
        case 0: 
            // Until robot has achieved an x translation of 1 m:
            if (robotMessage.x <= 1.0) {
                // Move in a straight line forward
                robotVelocity = 0.2;
                k = 0;
            } else {
                // Move on to next state
                state++;
            }
            break;

        case 1:
            // Until robot has achieved a 180 deg turn in theta:
            if (robotMessage.theta <= M_PI) {
                // Turn in a circle with radius 25 cm 
                robotVelocity = 0.2;
                k = 1 / 0.25;
            } else {
                state++;
            }
            break;

        case 2:
            // Until robot has achieved an x translation of -1 m:
            if (robotMessage.x >= 0) {
                // Move in a straight line forward
                robotVelocity = 0.2;
                k = 0;
            } else {
                // Move on to next state
                state++;
            }
            break;

        default: 
            // If not in any of the states, robot should just stop
            robotVelocity = 0;
            k = 0;
            break;
    }
    setWheelVelocities(robotVelocity, k);
    #endif

    #ifdef YOUR_TRAJECTORY
    // TODO: Create a state machine to define your custom trajectory!
    switch (state) {
        case 0: 
            // Until robot has achieved an x translation of 1 m:
            if (robotMessage.theta <= M_PI) {
                // Move in a straight line forward
                robotVelocity = 0.2;
                k = 1/0.1;
            } else {
                // Move on to next state
                state = state + 1;
            }
            break;

        case 1:
            // Until robot has achieved a 180 deg turn in theta:
            if (robotMessage.theta >= 0 ) {
                // Turn in a circle with radius 25 cm 
                robotVelocity = 0.2;
                k = - 1 / 0.1;
            } else {
                state = state - 1;
            }
            break;
        default: 
            // If not in any of the states, robot should just stop
            robotVelocity = 0;
            k = 0;
            break;
    }
    setWheelVelocities(robotVelocity, k);
    #endif 

}

void updateOdometry() {
    // Take angles from traction (rear) wheels only since they don't slip
    currPhiL = encoders[2].getPosition();
    currPhiR = -encoders[3].getPosition();
    
    // Update wheel angles and angular change
    double dPhiL = currPhiL - prevPhiL;
    double dPhiR = currPhiR - prevPhiR;
    prevPhiL = currPhiL;
    prevPhiR = currPhiR;

    // Calculate update in robot's base coordinates
    // float dtheta = r_wheel / (2 * b_base) * (dPhiR - dPhiL);
    gr = imu.getGyroReadings();
    float dtheta = gr.yawRate; 
    float dx = r_wheel / 2.0 * (cos(robotMessage.theta) * dPhiR + cos(robotMessage.theta) * dPhiL);
    float dy = r_wheel / 2.0 * (sin(robotMessage.theta) * dPhiR + sin(robotMessage.theta) * dPhiL);

    // Update robot message
    robotMessage.millis = millis();
    robotMessage.x += dx;
    robotMessage.y += dy;
    robotMessage.theta += dtheta;
}

