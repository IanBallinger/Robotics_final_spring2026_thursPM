#include <Bounce2.h>
#include <cstring>
#include <esp_now.h>
#include <WiFi.h>

#include "controller_pinout.h"
#include "display.h"
#include "dpad.h"
#include "joystick.h"
#include "util.h"
#include "wireless.h"

#define CONTROLLER_READ_PERIOD_MS 50
#define DEBUG_PRINT_PERIOD_MS 100
#define ROTARY_PRINT_PERIOD_MS 50
#define POT_DEADBAND 0.08f
#define COMMAND_FILTER_TAU_S 0.1f
#define ROTARY_SEESAW_ADDR 0x49

constexpr float PINCH_COMMAND_THRESHOLD = 0.5f;
constexpr float ELEVATOR_JOG_COMMAND_THRESHOLD = 0.35f;

enum class PinchState {
    NONE,
    OPEN,
    CLOSE,
};

static unsigned long last_command_filter_ms = 0;

static DPad rotaryDPad(ROTARY_SEESAW_ADDR);
static int32_t lastRotaryPosition = 0;

ControllerMessage prevControllerMessage;

static JoystickReading filteredLeftCommand = {0.0f, 0.0f};
static JoystickReading filteredRightCommand = {0.0f, 0.0f};
static PinchState lastPinchState = PinchState::NONE;

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
    return (int)mapDouble(
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

static const char* pinchStateToString(PinchState state) {
    switch (state) {
        case PinchState::OPEN:
            return "OPEN";
        case PinchState::CLOSE:
            return "CLOSE";
        default:
            return "NONE";
    }
}

static PinchState pinchStateFromJoystick(const JoystickReading& rightStick) {
    if (rightStick.y >= PINCH_COMMAND_THRESHOLD) {
        return PinchState::OPEN;
    }
    if (rightStick.y <= -PINCH_COMMAND_THRESHOLD) {
        return PinchState::CLOSE;
    }
    return PinchState::NONE;
}

static bool isZeroMac(const uint8_t* mac) {
    for (int i = 0; i < 6; ++i) {
        if (mac[i] != 0) {
            return false;
        }
    }
    return true;
}

static bool sendElevatorJogCommand(float jog_command) {
    if (isZeroMac(elevatorAddr)) {
        if (Serial) {
            Serial.println("ELV_JOG_TX,SKIP,UNCONFIGURED_MAC");
        }
        return false;
    }

    char msg[32];
    snprintf(msg, sizeof(msg), "ELV_JOG_CMD,%.3f", jog_command);
    const esp_err_t result = esp_now_send(
        elevatorAddr,
        reinterpret_cast<const uint8_t*>(msg),
        strlen(msg) + 1);

    if (Serial) {
        Serial.print("ELV_JOG_TX,");
        Serial.print(jog_command, 3);
        Serial.print(",");
        Serial.println(result == ESP_OK ? "OK" : "FAIL");
    }
    return result == ESP_OK;
}

static bool sendPinchCommand(PinchState state) {
    if (state == PinchState::NONE) {
        return false;
    }
    if (isZeroMac(elevatorAddr)) {
        if (Serial) {
            Serial.println("PINCH_TX,SKIP,UNCONFIGURED_MAC");
        }
        return false;
    }

    const char* pinch_cmd =
        (state == PinchState::OPEN) ? "PINCH_CMD,OPEN" : "PINCH_CMD,CLOSE";
    const esp_err_t result = esp_now_send(
        elevatorAddr,
        reinterpret_cast<const uint8_t*>(pinch_cmd),
        strlen(pinch_cmd) + 1);

    if (Serial) {
        Serial.print("PINCH_TX,");
        Serial.print(pinchStateToString(state));
        Serial.print(",");
        Serial.println(result == ESP_OK ? "OK" : "FAIL");
    }
    return result == ESP_OK;
}

static void printDebug(const JoystickReading& leftStick,
                       const JoystickReading& rightStick) {
    Serial.printf(
        "POTS raw_fb=%d raw_turn=%d mapped_fb=%.3f mapped_turn=%.3f joy1(x,y)=(%.3f, %.3f) joy2(x,y)=(%.3f, %.3f) buttons(L,R)=(%d,%d) pinch=%s\n",
        joystickRangeToAnalog(leftStick.y),
        joystickRangeToAnalog(rightStick.x),
        leftStick.y,
        rightStick.x,
        controllerMessage.joystick1.x,
        controllerMessage.joystick1.y,
        controllerMessage.joystick2.x,
        controllerMessage.joystick2.y,
        controllerMessage.buttonL ? 1 : 0,
        controllerMessage.buttonR ? 1 : 0,
        pinchStateToString(lastPinchState));
}

static void setupRotaryEncoderReadout() {
    rotaryDPad.setup();
    lastRotaryPosition = 0;
    Serial.printf("Rotary encoder ready on I2C addr 0x%02X\n", ROTARY_SEESAW_ADDR);
}

static void printRotaryEncoderReadout() {
    rotaryDPad.update();
    const DPadReading reading = rotaryDPad.read();
    const int32_t position = reading.encoderPosition;
    const int32_t delta = position - lastRotaryPosition;
    Serial.printf("ROTARY pos=%ld delta=%ld buttons(U,D,L,R,S)=(%d,%d,%d,%d,%d)\n",
                  static_cast<long>(position),
                  static_cast<long>(delta),
                  reading.up ? 1 : 0,
                  reading.down ? 1 : 0,
                  reading.left ? 1 : 0,
                  reading.right ? 1 : 0,
                  reading.select ? 1 : 0);
    lastRotaryPosition = position;
}

void setup() {
    Serial.begin(115200);

    setupWireless();

    joystick1.setup();
    joystick2.setup();
    pinMode(BUTTON_L_PIN, INPUT_PULLUP);
    pinMode(BUTTON_R_PIN, INPUT_PULLUP);

    memset(&controllerMessage, 0, sizeof(controllerMessage));
    memset(&prevControllerMessage, 0, sizeof(prevControllerMessage));

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

        // Shared controller packet mapping:
        // - joystick1.y -> forward/back for drive MCU
        // - joystick2.x -> turn for drive MCU and elevator jog source for elevator MCU
        // - joystick2.y -> pinch open/close source for elevator MCU
        controllerMessage.joystick1.x = filteredLeftCommand.x;
        controllerMessage.joystick1.y = filteredLeftCommand.y;
        controllerMessage.joystick2.x = filteredRightCommand.x;
        controllerMessage.joystick2.y = filteredRightCommand.y;

        controllerMessage.buttonL = (digitalRead(BUTTON_L_PIN) == LOW);
        controllerMessage.buttonR = (digitalRead(BUTTON_R_PIN) == LOW);

        const PinchState pinchState = pinchStateFromJoystick(filteredRightCommand);
        lastPinchState = pinchState;

        sendControllerData();
        prevControllerMessage = controllerMessage;
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(filteredLeftCommand, filteredRightCommand);
    }

}
