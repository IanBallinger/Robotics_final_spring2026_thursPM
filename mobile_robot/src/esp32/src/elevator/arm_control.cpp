#include "arm_control.h"

#include <algorithm>
#include <cmath>
#include <utility>

std::pair<float, float> inverseKinematics(
    double xE,
    double yE,
    double xO,
    double yO,
    double l1,
    double l2
) {
    if (l1 <= 0.0 || l2 <= 0.0) {
        return {NAN, NAN};
    }

    const double dx = xE - xO;
    const double dy = yE - yO;
    const double r2 = dx * dx + dy * dy;
    const double r = std::sqrt(r2);

    if (r <= 0.0) {
        return {NAN, NAN};
    }

    // Reachability check for a planar 2R arm.
    if (r > l1 + l2 || r < std::fabs(l1 - l2)) {
        return {NAN, NAN};
    }

    // Standard 2-link IK:
    // cos(theta2) = (r^2 - l1^2 - l2^2) / (2 l1 l2)
    // theta1 = atan2(dy, dx) - atan2(l2 sin(theta2), l1 + l2 cos(theta2))
    const double cos_theta2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    const double clamped_cos_theta2 = std::max(-1.0, std::min(1.0, cos_theta2));

    // Elbow-up branch.
    const double sin_theta2 = -std::sqrt(std::max(0.0, 1.0 - clamped_cos_theta2 * clamped_cos_theta2));
    const double theta2 = std::atan2(sin_theta2, clamped_cos_theta2);
    const double theta1 = std::atan2(dy, dx) - std::atan2(l2 * sin_theta2, l1 + l2 * clamped_cos_theta2);

    return {static_cast<float>(theta1), static_cast<float>(theta2)};
}
