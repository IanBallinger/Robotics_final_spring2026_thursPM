"""
Wire format for ``serial_to_from_jet.cpp`` on the ESP32.

Host → MCU (one line, newline-terminated)::

    WHL_CMD,<w1>,<w2>,<w3>,<w4>

Wheel ordering is canonical and must match the ESP32 wheel controller:
- w1 = left_front  (MOTOR 2)
- w2 = right_front (MOTOR 3)
- w3 = left_rear   (MOTOR 1)
- w4 = right_rear  (MOTOR 4)

MCU → host::

    ACK,<w1>,<w2>,<w3>,<w4>     # after a valid WHL_CMD (echo of commanded setpoints)
    IMU,<ax>,<ay>,<az>,<gx>,<gy>,<gz>   # periodic (~50 ms)
    WRONG_START | WRONG_NUM_VALUES     # bad WHL_CMD line
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

# --- Host → MCU -------------------------------------------------------------


def serialize_wheel_cmd(w1: float, w2: float, w3: float, w4: float) -> str:
    """Serialize a wheel command in canonical order ``(LF, RF, LR, RR)``."""
    return f"WHL_CMD,{w1},{w2},{w3},{w4}\n"


def serialize_arm_cmd(x_m: float, y_m: float) -> str:
    """Serialize a planar arm end-effector command in meters."""
    return f"ARM_CMD,{x_m},{y_m}\n"


# --- MCU → host -------------------------------------------------------------


@dataclass(frozen=True)
class IMUReading:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class EncoderReading:
    # Canonical wheel order from the ESP32 wheel controller:
    # w1 = left_front, w2 = right_front, w3 = left_rear, w4 = right_rear.
    w1: float
    w2: float
    w3: float
    w4: float


@dataclass(frozen=True)
class ArmXYCommand:
    x_m: float
    y_m: float


def deserialize_imu(line: str) -> Optional[IMUReading]:
    """
    Parse ``IMU,<ax>,<ay>,<az>,<gx>,<gy>,<gz>`` (optional trailing whitespace).
    Returns ``None`` if the line is not a valid IMU message.
    """
    s = line.strip()
    if not s.startswith("IMU,"):
        return None
    parts = s.split(",")
    if len(parts) != 7:
        return None
    try:
        ax, ay, az, gx, gy, gz = (float(parts[i]) for i in range(1, 7))
    except ValueError:
        return None
    return IMUReading(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)


def deserialize_encoder(line: str) -> Optional[EncoderReading]:
    """Parse ``ENC,<w1>,<w2>,<w3>,<w4>`` into measured wheel angular rates."""
    s = line.strip()
    if not s.startswith("ENC,"):
        return None
    parts = s.split(",")
    if len(parts) != 5:
        return None
    try:
        w1, w2, w3, w4 = (float(parts[i]) for i in range(1, 5))
    except ValueError:
        return None
    return EncoderReading(w1=w1, w2=w2, w3=w3, w4=w4)


def deserialize_arm_xy_command(line: str) -> Optional[ArmXYCommand]:
    """Parse ``ARM_CMD,<x_m>,<y_m>`` emitted by wheels joystick control."""
    s = line.strip()
    if not s.startswith("ARM_CMD,"):
        return None
    parts = s.split(",")
    if len(parts) != 3:
        return None
    try:
        x_m = float(parts[1])
        y_m = float(parts[2])
    except ValueError:
        return None
    return ArmXYCommand(x_m=x_m, y_m=y_m)


def parse_mcu_line(line: str) -> Optional[Union[IMUReading, EncoderReading, ArmXYCommand]]:
    return (
        deserialize_imu(line)
        or deserialize_encoder(line)
        or deserialize_arm_xy_command(line)
    )


__all__ = [
    "EncoderReading",
    "IMUReading",
    "ArmXYCommand",
    "serialize_arm_cmd",
    "serialize_wheel_cmd",
    "deserialize_encoder",
    "deserialize_imu",
    "deserialize_arm_xy_command",
    "parse_mcu_line",
]
