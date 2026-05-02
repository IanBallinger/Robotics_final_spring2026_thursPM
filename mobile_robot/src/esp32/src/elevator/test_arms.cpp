#include <ESP32Servo.h>

// P1 7, D2 3, P2 6

constexpr int ARM_SHOULDER_SERVO_PIN = 0;
constexpr int ARM_ELBOW_SERVO_PIN = 41;
constexpr int SERVO_MIN_US = 544;
constexpr int SERVO_MAX_US = 2400;

Servo shoulder_servo;
Servo elbow_servo;

void setup(){
    Serial.begin(115200);
    // shoulder_servo.attach(ARM_SHOULDER_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    elbow_servo.attach(ARM_ELBOW_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
}

void loop() {
  // Move to center position
  elbow_servo.write(90); 
  Serial.println("90");
  delay(1000);
  // Move to 0 degrees
  elbow_servo.write(0); 
  Serial.println("0");
  delay(1000);
  // Move to 180 degrees
  elbow_servo.write(180);
  Serial.println("180");
  delay(1000);
}