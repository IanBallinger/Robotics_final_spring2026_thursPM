#include <utility>

// Planar 2-link arm inverse kinematics.
// Returns {theta1, theta2} where:
// - theta1 is the shoulder angle in the world frame
// - theta2 is the elbow joint angle relative to link 1
//
// Coordinates:
// - (xO, yO) is the shoulder/base joint origin
// - (xE, yE) is the desired end-effector position
// - l1 and l2 are the link lengths
//
// The returned solution uses the elbow-up branch.
// If the target is unreachable, both values are NAN.
std::pair<float, float> inverseKinematics(
    double xE,
    double yE,
    double xO,
    double yO,
    double l1,
    double l2
);
