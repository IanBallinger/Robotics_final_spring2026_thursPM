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
#define COMMAND_FILTER_TAU_S 0.1f

static unsigned long last_command_filter_ms = 0;

ControllerMessage prevControllerMessage;

static JoystickReading filteredLeftCommand = {0.0f, 0.0f};
static JoystickReading filteredRightCommand = {0.0f, 0.0f};

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

static float lowPassValue(float currentValue, float targetValue, float alpha) {
    return currentValue + alpha * (targetValue - currentValue);
}

static JoystickReading lowPassReading(JoystickReading current,
                                      const JoystickReading& target,
                                      float alpha) {
    current.x = lowPassValue(current.x, target.x, alpha);
    current.y = lowPassValue(current.y, target.y, alpha);
    return current;
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
        const unsigned long now = millis();
        const float dt = (last_command_filter_ms == 0)
                             ? (CONTROLLER_READ_PERIOD_MS / 1000.0f)
                             : ((now - last_command_filter_ms) / 1000.0f);
        const float alpha = dt / (COMMAND_FILTER_TAU_S + dt);
        last_command_filter_ms = now;

        const JoystickReading leftStick = applyDeadband(joystick1.read());
        const JoystickReading rightStick = applyDeadband(joystick2.read());
        filteredLeftCommand = lowPassReading(filteredLeftCommand, leftStick, alpha);
        filteredRightCommand = lowPassReading(filteredRightCommand, rightStick, alpha);

        controllerMessage.millis = now;

        // left Y axis for forward/back motion
        controllerMessage.joystick1.y = filteredLeftCommand.y;

        // right X axis for left/right turning
        controllerMessage.joystick2.x = filteredRightCommand.x;

        // mirror turn onto joystick1.x for older single-stick code paths
        controllerMessage.joystick1.x = filteredRightCommand.x;
        controllerMessage.joystick2.y = 0.0f;

        if (!(prevControllerMessage == controllerMessage)) {
            sendControllerData();
            prevControllerMessage = controllerMessage;
        }
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(filteredLeftCommand, filteredRightCommand);
    }
}
