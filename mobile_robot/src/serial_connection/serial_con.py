"""
Serial link to the ESP32 firmware in ``esp32/src/robot/serial_to_from_jet.cpp``.

Default wire format:
- host -> MCU: ``WHL_CMD,w1,w2,w3,w4\n``
- MCU -> host: ``ACK,...`` / ``IMU,...`` / error lines

This transport intentionally rate-limits both directions:
- outbound wheel commands are buffered and only the latest pending command is sent
  at ``tx_rate_hz``
- inbound serial data is drained continuously on demand, but only the latest raw
  line / parsed IMU message is published to callers at ``rx_publish_rate_hz``

When ``debug`` is enabled, the transport prints the time between successive
published/sent messages.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple
import random

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
        timeout: float = 0.0,
        tx_rate_hz: float = 20.0,
        rx_publish_rate_hz: float = 20.0,
        debug: bool = False,
    ):
        self.port = port or os.environ.get("SERIAL_PORT", "/dev/ttyESP")
        self.baudrate = baudrate
        self._timeout = timeout
        self.tx_rate_hz = tx_rate_hz
        self.rx_publish_rate_hz = rx_publish_rate_hz
        self.debug = debug

        self._tx_period = 0.0 if tx_rate_hz <= 0 else 1.0 / tx_rate_hz
        self._rx_publish_period = 0.0 if rx_publish_rate_hz <= 0 else 1.0 / rx_publish_rate_hz

        self.ser = serial.Serial(port=self.port, baudrate=baudrate, timeout=timeout)
        self._rx_buffer = b""

        self._pending_wheel_cmd: Optional[Tuple[float, float, float, float]] = None
        self._last_tx_time: Optional[float] = None

        self._latest_raw_line: Optional[str] = None
        self._latest_raw_rx_time: Optional[float] = None
        self._last_raw_publish_time: Optional[float] = None
        self._last_raw_debug_time: Optional[float] = None

        self._latest_parsed_msg: Optional[IMUReading] = None
        self._latest_parsed_rx_time: Optional[float] = None
        self._last_parsed_publish_time: Optional[float] = None
        self._last_parsed_debug_time: Optional[float] = None

        if self.ser.is_open:
            print(
                f"Serial connected: {self.port} @ {baudrate} "
                f"(tx_rate_hz={tx_rate_hz}, rx_publish_rate_hz={rx_publish_rate_hz})"
            )

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "SerialConnect":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _debug_dt(self, tag: str, now: float, last: Optional[float]) -> None:
        if not self.debug:
            return
        if last is None:
            print(f"[{tag}] first message")
        else:
            print(f"[{tag}] dt={now - last:.6f}s")

    def _tx_due(self, now: float) -> bool:
        return (
            self._pending_wheel_cmd is not None
            and (
                self._tx_period <= 0.0
                or self._last_tx_time is None
                or (now - self._last_tx_time) >= self._tx_period
            )
        )

    def _publish_due(self, now: float, last_publish_time: Optional[float]) -> bool:
        return (
            self._rx_publish_period <= 0.0
            or last_publish_time is None
            or (now - last_publish_time) >= self._rx_publish_period
        )

    def _write_pending_if_due(self, force: bool = False) -> bool:
        now = time.monotonic()
        if self._pending_wheel_cmd is None:
            return False
        if not force and not self._tx_due(now):
            return False

        payload = serialize_wheel_cmd(*self._pending_wheel_cmd)
        self.ser.write(payload.encode("ascii"))
        self.ser.flush()
        self._debug_dt("TX WHL_CMD", now, self._last_tx_time)
        if self.debug:
            print(f"[TX WHL_CMD] {payload.strip()}")
        self._last_tx_time = now
        self._pending_wheel_cmd = None
        return True

    def send_wheel_cmd(
        self,
        w1: float,
        w2: float,
        w3: float = 0.0,
        w4: float = 0.0,
        force: bool = False,
    ) -> bool:
        """
        Buffer the latest wheel command and transmit at the configured fixed rate.
        Returns True only when a command is actually written to the serial port.
        """
        self._pending_wheel_cmd = (w1, w2, w3, w4)
        return self._write_pending_if_due(force=force)

    def flush_tx(self, force: bool = False) -> bool:
        """Attempt to send the latest pending buffered command."""
        return self._write_pending_if_due(force=force)

    def readline(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Compatibility helper for one blocking/non-blocking raw line read.
        Most callers should prefer ``read`` / ``read_parsed``.
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

    def poll(self, max_lines: int = 256) -> int:
        """
        Drain currently available serial bytes, parse complete newline-terminated
        records, and keep only the latest raw line / parsed IMU message.
        Returns the number of complete lines consumed from the port.
        """
        consumed = 0
        waiting = self.ser.in_waiting
        if waiting > 0:
            self._rx_buffer += self.ser.read(waiting)

        while consumed < max_lines:
            newline_idx = self._rx_buffer.find(b"\n")
            if newline_idx < 0:
                break
            raw_line = self._rx_buffer[:newline_idx]
            self._rx_buffer = self._rx_buffer[newline_idx + 1 :]

            line = raw_line.decode("ascii", errors="replace").strip()
            consumed += 1
            if not line:
                continue

            now = time.monotonic()
            self._latest_raw_line = line
            self._latest_raw_rx_time = now

            parsed = parse_mcu_line(line)
            if parsed is not None:
                self._latest_parsed_msg = parsed
                self._latest_parsed_rx_time = now

        return consumed

    def read(self, max_lines: int = 256) -> List[str]:
        """
        Drain the RX buffer, but publish at most the latest raw line at the
        configured fixed output rate.
        """
        self.poll(max_lines=max_lines)
        if self._latest_raw_line is None or self._latest_raw_rx_time is None:
            return []

        now = time.monotonic()
        if not self._publish_due(now, self._last_raw_publish_time):
            return []

        line = self._latest_raw_line
        self._last_raw_publish_time = now
        self._latest_raw_line = None
        self._debug_dt("RX RAW", now, self._last_raw_debug_time)
        if self.debug:
            age = now - self._latest_raw_rx_time
            print(f"[RX RAW] age={age:.6f}s line={line}")
        self._last_raw_debug_time = now
        self._latest_raw_rx_time = None
        return [line]

    def read_parsed(self, max_lines: int = 256) -> List[IMUReading]:
        """
        Drain the RX buffer, but publish at most the latest parsed IMU message at
        the configured fixed output rate.
        """
        self.poll(max_lines=max_lines)
        if self._latest_parsed_msg is None or self._latest_parsed_rx_time is None:
            return []

        now = time.monotonic()
        if not self._publish_due(now, self._last_parsed_publish_time):
            return []

        msg = self._latest_parsed_msg
        self._last_parsed_publish_time = now
        self._latest_parsed_msg = None
        self._debug_dt("RX IMU", now, self._last_parsed_debug_time)
        if self.debug:
            age = now - self._latest_parsed_rx_time
            print(f"[RX IMU] age={age:.6f}s msg={msg}")
        self._last_parsed_debug_time = now
        self._latest_parsed_rx_time = None
        return [msg]


if __name__ == "__main__":
    port = "/dev/ttyESP_WHL"
    con = SerialConnect(port=port, tx_rate_hz=2.0, rx_publish_rate_hz=2.0, debug=True)
    try:
        while True:
            w1 = random.uniform(-1.0, 1.0)
            w2 = random.uniform(-1.0, 1.0)
            w3 = random.uniform(-1.0, 1.0)
            w4 = random.uniform(-1.0, 1.0)
            con.send_wheel_cmd(w1, w2, w3, w4)
            con.flush_tx()
            for msg in con.read_parsed(max_lines=64):
                print(msg)
            time.sleep(0.01)
    finally:
        con.close()
