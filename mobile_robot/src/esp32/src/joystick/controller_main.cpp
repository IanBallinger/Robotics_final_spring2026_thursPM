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
Joystick joystick1(9, 6);
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
        "POTS raw_fb=%d raw_turn=%d mapped_fb=%.3f mapped_turn=%.3f joy1(x,y)=(%.3f, %.3f) joy2(x,y)=(%.3f, %.3f) buttons(L,R)=(%d,%d)\n",
        joystickRangeToAnalog(leftStick.y),
        joystickRangeToAnalog(rightStick.y),
        leftStick.y,
        rightStick.y,
        controllerMessage.joystick1.x,
        controllerMessage.joystick1.y,
        controllerMessage.joystick2.x,
        controllerMessage.joystick2.y,
        controllerMessage.buttonL ? 1 : 0,
        controllerMessage.buttonR ? 1 : 0);
}

void setup() {
    Serial.begin(115200);

    setupWireless();

    joystick1.setup();
    joystick2.setup();

    // Active-low buttons (pressed = LOW). Logical true in ControllerMessage = pressed.
    pinMode(BUTTON_L_PIN, INPUT_PULLUP);
    pinMode(BUTTON_R_PIN, INPUT_PULLUP);

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

        // Differential-drive mapping:
        // - left-stick forward/back comes from joystick1.y
        // - right-stick up/down comes from joystick2.y and commands turn-in-place
        controllerMessage.joystick1.x = filteredLeftCommand.x;
        controllerMessage.joystick1.y = filteredLeftCommand.y;
        controllerMessage.joystick2.x = filteredRightCommand.x;
        controllerMessage.joystick2.y = filteredRightCommand.y;

        // Remote: right button toggles manual/autonomy on the drive ESP32 (see
        // serial_to_from_jet.cpp). Left button is available for future use.
        controllerMessage.buttonL = (digitalRead(BUTTON_L_PIN) == LOW);
        controllerMessage.buttonR = (digitalRead(BUTTON_R_PIN) == LOW);

        if (!(prevControllerMessage == controllerMessage)) {
            sendControllerData();
            prevControllerMessage = controllerMessage;
        }
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(filteredLeftCommand, filteredRightCommand);
    }
}
