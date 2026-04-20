#ifndef CONTROLLER_PINOUT_H
#define CONTROLLER_PINOUT_H

#define JOYSTICK1_X_PIN A0
#define JOYSTICK1_Y_PIN A1
#define JOYSTICK2_X_PIN A2
#define JOYSTICK2_Y_PIN A3

// Two-potentiometer controller inputs:
// - left Y axis  -> forward/backward speed
// - right X axis -> left/right turning
#define POT_FORWARD_BACK_PIN JOYSTICK1_Y_PIN
#define POT_TURN_PIN JOYSTICK2_X_PIN
#define BUTTON_L_PIN 16
#define BUTTON_R_PIN 39
#define TFT_CS_PIN 12
#define TFT_DC_PIN 13

#endif // CONTROLLER_PINOUT_H