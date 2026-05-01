#include <ESP32Servo.h>

// P1 41, D2 40, P2 42

constexpr int ARM_SHOULDER_SERVO_PIN = 40;
constexpr int ARM_ELBOW_SERVO_PIN = 41;
constexpr int SERVO_MIN_US = 544;
constexpr int SERVO_MAX_US = 2400;

Servo shoulder_servo;
Servo elbow_servo;

void setup(){
    Serial.begin(115200);
    shoulder_servo.attach(ARM_SHOULDER_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    // elbow_servo.attach(ARM_ELBOW_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
}

void loop() {
  // Move to center position
  shoulder_servo.write(90); 
  Serial.println("90");
  delay(1000);
  // Move to 0 degrees
  shoulder_servo.write(0); 
  Serial.println("0");
  delay(1000);
  // Move to 180 degrees
  shoulder_servo.write(180);
  Serial.println("180");
  delay(1000);
}