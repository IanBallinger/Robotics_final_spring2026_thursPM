"""
Wire format for ``serial_to_from_jet.cpp`` on the ESP32.

Host → MCU (one line, newline-terminated)::

    WHL_CMD,<w1>,<w2>,<w3>,<w4>

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
    """Same as ``serialize_whl`` with explicit wheel indices."""
    return f"WHL_CMD,{w1},{w2},{w3},{w4}\n"


# --- MCU → host -------------------------------------------------------------


@dataclass(frozen=True)
class IMUReading:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


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


def parse_mcu_line(line: str) -> Optional[IMUReading]:
    return deserialize_imu(line)


__all__ = [
    "IMUReading",
    "serialize_wheel_cmd",
    "deserialize_imu",
    "parse_mcu_line",
]
