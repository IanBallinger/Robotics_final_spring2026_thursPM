"""
Wire format for ``src/esp32/src/elevator/elevator_control.cpp`` on the ESP32.

Host → MCU (one line, newline-terminated)::

    ELV_CMD,<desired_height_m>

MCU → host::

    ELV_ACK,<desired_height_m>    # latest applied commanded height
    ELV_MEAS,<height_m>           # periodic elevator height measurement
    WRONG_START | WRONG_NUM_VALUES
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ElevatorHeightReading:
    height_m: float


def serialize_elevator_cmd(desired_height_m: float) -> str:
    return f"ELV_CMD,{desired_height_m}\n"


def deserialize_elevator_height(line: str) -> Optional[ElevatorHeightReading]:
    s = line.strip()
    if not s.startswith("ELV_MEAS,"):
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        height_m = float(parts[1])
    except ValueError:
        return None
    return ElevatorHeightReading(height_m=height_m)


def parse_elevator_line(line: str) -> Optional[ElevatorHeightReading]:
    return deserialize_elevator_height(line)


__all__ = [
    "ElevatorHeightReading",
    "serialize_elevator_cmd",
    "deserialize_elevator_height",
    "parse_elevator_line",
]
