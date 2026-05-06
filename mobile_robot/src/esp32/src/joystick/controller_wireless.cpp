#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "wireless.h"

// #define PRINT_CONTROLLER
#define PRINT_ROBOT

const uint8_t* peerAddr = robotAddr;
esp_now_peer_info_t peerInfo;

bool freshWirelessData = false;
ControllerMessage controllerMessage;
RobotMessage robotMessage;

void onSendData(const uint8_t* mac_addr, esp_now_send_status_t status) {
    (void)mac_addr;
    bool success = status == ESP_NOW_SEND_SUCCESS;
    if (success && Serial) {
        Serial.println("Sent");
#ifdef PRINT_CONTROLLER
        controllerMessage.print();
#endif
    } else if (Serial) {
        Serial.println("Failed");
    }
}

void onRecvData(const uint8_t* mac, const uint8_t* incomingData, int len) {
    (void)mac;
    (void)len;
    memcpy(&robotMessage, incomingData, sizeof(robotMessage));
    freshWirelessData = true;
#ifdef PRINT_ROBOT
    if (Serial) robotMessage.print();
#endif
}

static bool isZeroMac(const uint8_t* mac) {
    for (int i = 0; i < 6; ++i) {
        if (mac[i] != 0) {
            return false;
        }
    }
    return true;
}

bool sendControllerData() {
    const esp_err_t robot_result = esp_now_send(robotAddr, (uint8_t*)&controllerMessage, sizeof(controllerMessage));
    esp_err_t elevator_result = ESP_OK;
    if (!isZeroMac(elevatorAddr)) {
        elevator_result = esp_now_send(elevatorAddr, (uint8_t*)&controllerMessage, sizeof(controllerMessage));
    }
    return robot_result == ESP_OK && elevator_result == ESP_OK;
}
