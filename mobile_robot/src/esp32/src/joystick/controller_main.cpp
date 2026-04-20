#include <Bounce2.h>
#include "wireless.h"
#include "util.h"
#include "joystick.h"
#include "dpad.h"
#include "display.h"
#include "controller_pinout.h"

#define CONTROLLER_READ_PERIOD_MS 50
#define DEBUG_PRINT_PERIOD_MS 100
#define POT_DEADBAND 0.08f

ControllerMessage prevControllerMessage;

// Use the original joystick abstraction/setup.
Joystick joystick1(34, 6);
Joystick joystick2(1, 5);

static float applyDeadband(float value, float deadband) {
    return (abs(value) < deadband) ? 0.0f : value;
}

static JoystickReading applyDeadband(const JoystickReading& reading) {
    return {
        applyDeadband(reading.x, POT_DEADBAND),
        applyDeadband(reading.y, POT_DEADBAND)
    };
}

static int joystickRangeToAnalog(float value) {
    return (int) mapDouble(
        value,
        JOYSTICK_READING_MIN,
        JOYSTICK_READING_MAX,
        JOYSTICK_ANALOG_MIN,
        JOYSTICK_ANALOG_MAX);
}

static void printDebug(const JoystickReading& leftStick, const JoystickReading& rightStick) {
    Serial.printf(
        "POTS raw_fb=%d raw_turn=%d mapped_fb=%.3f mapped_turn=%.3f joy1(x,y)=(%.3f, %.3f) joy2(x,y)=(%.3f, %.3f)\n",
        joystickRangeToAnalog(leftStick.y),
        joystickRangeToAnalog(rightStick.x),
        leftStick.y,
        rightStick.x,
        controllerMessage.joystick1.x,
        controllerMessage.joystick1.y,
        controllerMessage.joystick2.x,
        controllerMessage.joystick2.y);
}

void setup() {
    Serial.begin(115200);

    setupWireless();

    joystick1.setup();
    joystick2.setup();

    Serial.println("Two-pot joystick app ready.");
}

void loop() {
    EVERY_N_MILLIS(CONTROLLER_READ_PERIOD_MS) {
        const JoystickReading leftStick = applyDeadband(joystick1.read());
        const JoystickReading rightStick = applyDeadband(joystick2.read());

        controllerMessage.millis = millis();

        // left Y axis for forward/back motion
        controllerMessage.joystick1.y = leftStick.y;

        // right X axis for left/right turning
        controllerMessage.joystick2.x = rightStick.x;

        // mirror turn onto joystick1.x for older single-stick code paths
        controllerMessage.joystick1.x = rightStick.x;
        controllerMessage.joystick2.y = 0.0f;

        if (!(prevControllerMessage == controllerMessage)) {
            sendControllerData();
            prevControllerMessage = controllerMessage;
        }
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(controllerMessage.joystick1, controllerMessage.joystick2);
    }
}
