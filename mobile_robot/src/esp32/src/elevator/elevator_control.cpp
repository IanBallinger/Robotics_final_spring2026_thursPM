#include "Adafruit_VL53L0X.h"
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#define TCAADDR 0x70

struct DesiredElevatorState {
  float height_m;

  DesiredElevatorState() : height_m(0.0f) {}
  explicit DesiredElevatorState(float height_m_) : height_m(height_m_) {}
};

Adafruit_VL53L0X lox = Adafruit_VL53L0X();
String rx_line = "";

constexpr uint8_t TOF_MUX_CHANNEL = 0;
constexpr unsigned long CMD_APPLY_PERIOD_MS = 50;
constexpr unsigned long MEAS_PUBLISH_PERIOD_MS = 50;
constexpr unsigned long ACK_PUBLISH_PERIOD_MS = 50;
constexpr unsigned long CMD_TIMEOUT_MS = 250;
constexpr bool SERIAL_DEBUG_TIMING = true;
constexpr float MM_TO_M = 1.0f / 1000.0f;

DesiredElevatorState latest_rx_cmd;
DesiredElevatorState latest_applied_cmd;
bool has_pending_cmd = false;
bool ack_dirty = false;
unsigned long last_cmd_rx_ms = 0;
unsigned long last_cmd_apply_ms = 0;
unsigned long last_meas_publish_ms = 0;
unsigned long last_ack_publish_ms = 0;
unsigned long last_ack_debug_ms = 0;
unsigned long last_meas_debug_ms = 0;

void tcaSelect(uint8_t channel) {
  if (channel > 7) {
    return;
  }
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

static void printDebugTiming(const char* tag, unsigned long& last_ms) {
  if (!SERIAL_DEBUG_TIMING) {
    return;
  }

  const unsigned long now = millis();
  Serial.print("DBG,");
  Serial.print(tag);
  Serial.print(",dt_ms,");
  if (last_ms == 0) {
    Serial.println("FIRST");
  } else {
    Serial.println(now - last_ms);
  }
  last_ms = now;
}

static bool handleElevatorCommand(const String& line, DesiredElevatorState& cmd) {
  if (!line.startsWith("ELV_CMD,")) {
    Serial.println("WRONG_START");
    return false;
  }

  if (sscanf(line.c_str(), "ELV_CMD,%f", &cmd.height_m) != 1) {
    Serial.println("WRONG_NUM_VALUES");
    return false;
  }

  return true;
}

static float readElevatorHeightMeters() {
  tcaSelect(TOF_MUX_CHANNEL);
  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus == 4) {
    return NAN;
  }

  return static_cast<float>(measure.RangeMilliMeter) * MM_TO_M;
}

static void applyElevatorCommand(const DesiredElevatorState& cmd,
                                 float measured_height_m) {
  latest_applied_cmd = cmd;

  // TODO: replace this placeholder with actual elevator motor control.
  // Example closed-loop quantity to drive the motor with:
  //   const float error_m = cmd.height_m - measured_height_m;
  //   motor.drive(pid.calculate(measured_height_m, cmd.height_m));
  (void)measured_height_m;
}

static void printElevatorAck(const DesiredElevatorState& cmd) {
  printDebugTiming("ELV_ACK", last_ack_debug_ms);
  Serial.print("ELV_ACK,");
  Serial.println(cmd.height_m, 4);
}

static void publishElevatorMeasurement(float height_m) {
  printDebugTiming("ELV_MEAS", last_meas_debug_ms);
  Serial.print("ELV_MEAS,");
  Serial.println(height_m, 4);
}

static void stopElevator() {
  latest_rx_cmd = DesiredElevatorState();
  latest_applied_cmd = DesiredElevatorState();
  has_pending_cmd = false;
  ack_dirty = false;

  // TODO: stop the real elevator motor driver here.
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  tcaSelect(TOF_MUX_CHANNEL);
  if (!lox.begin()) {
    Serial.println("ERR,VL53L0X_INIT_FAILED");
    while (1) {
      delay(100);
    }
  }

  const unsigned long now = millis();
  last_cmd_rx_ms = now;
  last_cmd_apply_ms = now;
  last_meas_publish_ms = now;
  last_ack_publish_ms = now;

  Serial.println("ELV_READY");
}

void loop() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      rx_line.trim();
      DesiredElevatorState cmd;
      if (rx_line.length() > 0 && handleElevatorCommand(rx_line, cmd)) {
        latest_rx_cmd = cmd;
        has_pending_cmd = true;
        last_cmd_rx_ms = millis();
      }
      rx_line = "";
    } else {
      rx_line += c;
    }
  }

  const unsigned long now = millis();
  const float measured_height_m = readElevatorHeightMeters();

  if (now - last_cmd_rx_ms > CMD_TIMEOUT_MS) {
    stopElevator();
  }

  if (has_pending_cmd && now - last_cmd_apply_ms >= CMD_APPLY_PERIOD_MS) {
    applyElevatorCommand(latest_rx_cmd, measured_height_m);
    ack_dirty = true;
    has_pending_cmd = false;
    last_cmd_apply_ms = now;
  }

  if (ack_dirty && now - last_ack_publish_ms >= ACK_PUBLISH_PERIOD_MS) {
    printElevatorAck(latest_applied_cmd);
    ack_dirty = false;
    last_ack_publish_ms = now;
  }

  if (!isnan(measured_height_m) && now - last_meas_publish_ms >= MEAS_PUBLISH_PERIOD_MS) {
    publishElevatorMeasurement(measured_height_m);
    last_meas_publish_ms = now;
  }
}
