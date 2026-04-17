"""
Serial link to the ESP32 firmware in ``esp32/src/robot/serial_to_from_jet.cpp``.

Default: 115200 baud, ``WHL_CMD,...`` outbound; inbound  ``IMU`` lines are parsed by ``serialization.parse_mcu_line``.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import serial

try:
    from .serialization import (
        IMUReading,
        serialize_wheel_cmd,
        parse_mcu_line,
    )
except ImportError:
    from serialization import (  # type: ignore[no-redef]
        IMUReading,
        serialize_wheel_cmd,
        parse_mcu_line,
    )


class SerialConnect:
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        self.port = port or os.environ.get("SERIAL_PORT", "/dev/ttyESP")
        self.baudrate = baudrate
        self._timeout = timeout
        self.ser = serial.Serial(port=self.port, baudrate=baudrate, timeout=timeout)
        if self.ser.is_open:
            print(f"Serial connected: {self.port} @ {baudrate}")

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "SerialConnect":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def send_wheel_cmd(
        self, w1: float, w2: float, w3: float = 0.0, w4: float = 0.0
    ) -> None:
        """Send ``WHL_CMD,<w1>,<w2>,<w3>,<w4>`` (matches MCU ``sscanf`` format)."""
        data_str = serialize_wheel_cmd(w1, w2, w3, w4)
        self.ser.write(data_str.encode("ascii"))
        self.ser.flush()

    def readline(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Read one newline-terminated line (stripped). Returns ``None`` on timeout
        or empty read.
        """
        old = self.ser.timeout
        try:
            if timeout is not None:
                self.ser.timeout = timeout
            raw = self.ser.readline()
            if not raw:
                return None
            return raw.decode("ascii", errors="replace").strip()
        finally:
            self.ser.timeout = old

    def read(self, max_lines: int = 32) -> List[str]:
        """
        Drain up to ``max_lines`` complete lines currently in the RX buffer.
        Uses the port's configured timeout per line.
        """
        lines: List[str] = []
        for _ in range(max_lines):
            line = self.readline()
            if line is None or line == "":
                break
            lines.append(line)
        return lines

    def read_parsed(self, max_lines: int = 32) -> List[IMUReading]:
        """Like ``read`` but each line is passed through ``parse_mcu_line`` (drops unknown)."""
        out: List[IMUReading] = []
        for line in self.read(max_lines=max_lines):
            msg = parse_mcu_line(line)
            if msg is not None:
                out.append(msg)
        return out


if __name__ == "__main__":
    con = SerialConnect()
    try:
        while True:
            con.send_wheel_cmd(0.0, 0.0, 0.0, 0.0)
            for msg in con.read_parsed(max_lines=8):
                print(msg)
            time.sleep(0.5)
    finally:
        con.close()
