#ifndef PINCH_WIRELESS_H
#define PINCH_WIRELESS_H

#include <stdint.h>

/** Must match wireless.h `controllerAddr` (handheld transmitter). */
extern const uint8_t pinch_controller_addr[6];

/**
 * Set to your elevator ESP32 Wi‑Fi STA MAC (same order as WiFi.macAddress()).
 * Leave first byte 0xFF to disable controller→elevator pinch packets (controller
 * will not add the ESP-NOW peer or send).
 */
extern const uint8_t elevatorAddr[6];

#define PINCH_ESPNOW_VERSION 1u

/** Compact packet: wheel MCU ignores non-ControllerMessage lengths. */
struct PinchEspNowPacket {
  uint8_t version;
  /** -1 = pinch closed, 0 = encoder neutral / hold last target, 1 = pinch open */
  int8_t mode;
};

#endif
