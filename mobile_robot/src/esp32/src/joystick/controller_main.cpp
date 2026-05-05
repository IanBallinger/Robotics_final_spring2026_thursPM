#include <Bounce2.h>
#include <cstring>
#include <esp_now.h>
#include <WiFi.h>

#include "../../include/pinch_wireless.h"
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

enum class PinchState {
    NONE,
    OPEN,
    CLOSE,
};

// ANO / seesaw encoder counts: rotate right -> open, left -> close (flip sign if reversed).
#ifndef PINCH_ENCODER_OPEN_TICKS
#define PINCH_ENCODER_OPEN_TICKS 1
#endif
#ifndef PINCH_ENCODER_CLOSE_TICKS
#define PINCH_ENCODER_CLOSE_TICKS (-1)
#endif

static unsigned long last_command_filter_ms = 0;

static DPad rotaryDPad(ROTARY_SEESAW_ADDR);
static int32_t lastRotaryPosition = 0;

ControllerMessage prevControllerMessage;

static JoystickReading filteredLeftCommand = {0.0f, 0.0f};
static JoystickReading filteredRightCommand = {0.0f, 0.0f};
static PinchState lastPinchState = PinchState::NONE;

// Use the original joystick abstraction/setup.
Joystick joystick1(9, 6);
Joystick joystick2(1, 5);

// ANO Rotary Encoder with Adafruit Seesaw (default I2C 0x49); same stack as DPad class.
static DPad dpad(0x49);
static int8_t last_pinch_mode_sent = 127;

static int8_t pinchModeFromEncoder(int32_t encoder_ticks) {
#ifdef PINCH_ENCODER_INVERT
  encoder_ticks = -encoder_ticks;
#endif
  if (encoder_ticks >= PINCH_ENCODER_OPEN_TICKS) {
    return 1;
  }
  if (encoder_ticks <= PINCH_ENCODER_CLOSE_TICKS) {
    return -1;
  }
  return 0;
}

static void addElevatorPinchPeerIfEnabled() {
  if (elevatorAddr[0] == static_cast<uint8_t>(0xFF)) {
    return;
  }
  esp_now_peer_info_t peer{};
  memcpy(peer.peer_addr, elevatorAddr, 6);
  peer.channel = 0;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("WARN,PINCH_ESPNOW_PEER_FAIL");
  }
}

static void sendPinchPacketIfChanged(int8_t mode, int8_t* last_sent) {
  if (elevatorAddr[0] == static_cast<uint8_t>(0xFF)) {
    return;
  }
  if (mode == *last_sent) {
    return;
  }
  PinchEspNowPacket pkt;
  pkt.version = PINCH_ESPNOW_VERSION;
  pkt.mode = mode;
  esp_now_send(elevatorAddr, reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
  *last_sent = mode;
}

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

static bool sendPinchCommand(PinchState state) {
    if (state == PinchState::NONE) {
        return false;
    }

    const char* pinch_cmd = (state == PinchState::OPEN) ? "PINCH_CMD,OPEN" : "PINCH_CMD,CLOSE";
    const esp_err_t result = esp_now_send(elevatorAddr,
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

static void printDebug(const JoystickReading& leftStick, const JoystickReading& rightStick) {
    Serial.printf(
        "POTS raw_fb=%d raw_turn=%d mapped_fb=%.3f mapped_turn=%.3f joy1(x,y)=(%.3f, %.3f) joy2(x,y)=(%.3f, %.3f) buttons(L,R)=(%d,%d) pinch=%s\n",
        joystickRangeToAnalog(leftStick.y),
        joystickRangeToAnalog(rightStick.y),
        leftStick.y,
        rightStick.y,
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
<<<<<<< HEAD
    setupRotaryEncoderReadout();
=======
    addElevatorPinchPeerIfEnabled();
>>>>>>> 7d44ee2 (added joystick pinch and serial pinch commands)

    joystick1.setup();
    joystick2.setup();
    dpad.setup();

    // Active-low buttons (pressed = LOW). Logical true in ControllerMessage = pressed.
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

        dpad.update();
        const DPadReading dpad_reading = dpad.read(false);
        controllerMessage.dPad = dpad_reading;

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

<<<<<<< HEAD
        const PinchState pinchState = pinchStateFromJoystick(filteredRightCommand);
        if (pinchState != lastPinchState) {
            sendPinchCommand(pinchState);
            lastPinchState = pinchState;
        }
=======
        const int8_t pinch_mode = pinchModeFromEncoder(dpad_reading.encoderPosition);
        sendPinchPacketIfChanged(pinch_mode, &last_pinch_mode_sent);
>>>>>>> 7d44ee2 (added joystick pinch and serial pinch commands)

        if (!(prevControllerMessage == controllerMessage)) {
            sendControllerData();
            prevControllerMessage = controllerMessage;
        }
    }

    EVERY_N_MILLIS(DEBUG_PRINT_PERIOD_MS) {
        printDebug(filteredLeftCommand, filteredRightCommand);
    }

    EVERY_N_MILLIS(ROTARY_PRINT_PERIOD_MS) {
        printRotaryEncoderReadout();
    }
}