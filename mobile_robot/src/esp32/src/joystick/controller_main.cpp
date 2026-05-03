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
#define BUTTON_ACTIVE_LOW true

// Mirror wheels-side arm mapping for operator debug visibility.
#define ARM_JOYSTICK_DEADBAND 0.1f
#define ARM_MIN_RADIUS_M 0.10f
#define ARM_MAX_RADIUS_M 0.42f
#define ARM_X_CENTER_M 0.30f
#define ARM_Y_CENTER_M 0.10f
#define ARM_MAX_XY_SPEED_MPS 0.08f
#define CONTROLLER_DT_S (CONTROLLER_READ_PERIOD_MS / 1000.0f)

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

static bool isButtonPressed(int pin) {
    const int level = digitalRead(pin);
    return BUTTON_ACTIVE_LOW ? (level == LOW) : (level == HIGH);
}

static JoystickReading lowPassReading(JoystickReading current,
                                      const JoystickReading& target,
                                      float alpha) {
    current.x = lowPassValue(current.x, target.x, alpha);
    current.y = lowPassValue(current.y, target.y, alpha);
    return current;
}

static void estimateArmDeltaForDebug(const JoystickReading& rightStick,
                                     float& armDx,
                                     float& armDy) {
    float x_input = rightStick.x;
    float y_input = rightStick.y;

    if (fabs(x_input) < ARM_JOYSTICK_DEADBAND) {
        x_input = 0.0f;
    }
    if (fabs(y_input) < ARM_JOYSTICK_DEADBAND) {
        y_input = 0.0f;
    }

    armDx = x_input * ARM_MAX_XY_SPEED_MPS * CONTROLLER_DT_S;
    armDy = y_input * ARM_MAX_XY_SPEED_MPS * CONTROLLER_DT_S;
}

static void printDebug(const JoystickReading& leftStick, const JoystickReading& rightStick) {
    float armDx = 0.0f;
    float armDy = 0.0f;
    estimateArmDeltaForDebug(rightStick, armDx, armDy);
    const int rawBtnRLevel = digitalRead(BUTTON_R_PIN);

    Serial.printf(
    "POTS raw_fb=%d raw_turn=%d mapped_fb=%.3f mapped_turn=%.3f btnR=%d raw_btnR=%d joy1(x,y)=(%.3f, %.3f) joy2(x,y)=(%.3f, %.3f) arm_dxy=(%.4f, %.4f)\n",
        joystickRangeToAnalog(leftStick.y),
        joystickRangeToAnalog(leftStick.x),
        leftStick.y,
        leftStick.x,
        controllerMessage.buttonR,
        rawBtnRLevel,
        controllerMessage.joystick1.x,
        controllerMessage.joystick1.y,
        controllerMessage.joystick2.x,
        controllerMessage.joystick2.y,
        armDx,
        armDy);
}

void setup() {
    Serial.begin(115200);

    setupWireless();

    pinMode(BUTTON_L_PIN, INPUT_PULLUP);
    pinMode(BUTTON_R_PIN, INPUT);

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

        // Differential-drive mapping:
        // - left-stick forward/back comes from joystick1.y
        // - left-stick left/right comes from joystick1.x and commands turn-in-place
        controllerMessage.joystick1.x = filteredLeftCommand.x;
        controllerMessage.joystick1.y = filteredLeftCommand.y;
        controllerMessage.joystick2.x = filteredRightCommand.x;
        controllerMessage.joystick2.y = filteredRightCommand.y;
        controllerMessage.buttonL = isButtonPressed(BUTTON_L_PIN);
        controllerMessage.buttonR = isButtonPressed(BUTTON_R_PIN);

        if (!(prevControllerMessage == controllerMessage)) {
            sendControllerData();
            prevControllerMessage = controllerMessage;
        }
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(filteredLeftCommand, filteredRightCommand);
    }
}
