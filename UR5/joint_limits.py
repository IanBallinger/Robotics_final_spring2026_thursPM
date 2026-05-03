"""Canonical UR5 joint limit definitions for this workspace.

This module intentionally has no RTDE/OpenCV dependencies so non-robot tools
(e.g. Julia visualizers) can import it in offline environments.
"""

from math import pi

# Format: [(lower_rad, upper_rad), ...] for joints q0..q5.
# Keep this as the single source of truth for operator-facing range monitors.
UR5_JOINT_LIMITS_RAD = [
    (-2.0 * pi, 2.0 * pi),  # q0 base
    (-2.0 * pi, 2.0 * pi),  # q1 shoulder
    (-2.0 * pi, 2.0 * pi),  # q2 elbow
    (-2.0 * pi, 2.0 * pi),  # q3 wrist1
    (-2.0 * pi, 2.0 * pi),  # q4 wrist2
    (-2.0 * pi, 2.0 * pi),  # q5 wrist3
]


def get_joint_limits_rad():
    """Return a copy of configured joint limits in radians."""
    return list(UR5_JOINT_LIMITS_RAD)
