from __future__ import annotations

import os
import time
from typing import List, Optional

import serial

try:
    from .elevator_serialization import (
        ElevatorHeightReading,
        parse_elevator_line,
        serialize_elevator_cmd,
        serialize_pinch_cmd,
    )
    from .serialization import serialize_arm_cmd
except ImportError:
    from elevator_serialization import (  # type: ignore[no-redef]
        ElevatorHeightReading,
        parse_elevator_line,
        serialize_elevator_cmd,
        serialize_pinch_cmd,
    )
    from serialization import serialize_arm_cmd  # type: ignore[no-redef]


class ElevatorSerialConnect:
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 0.0,
        tx_rate_hz: float = 20.0,
        rx_publish_rate_hz: float = 20.0,
        debug: bool = False,
    ):
        self.port = port or os.environ.get("ELEVATOR_SERIAL_PORT", "/dev/ttyESP_ELV")
        self.baudrate = baudrate
        self._timeout = timeout
        self.tx_rate_hz = tx_rate_hz
        self.rx_publish_rate_hz = rx_publish_rate_hz
        self.debug = debug

        self._tx_period = 0.0 if tx_rate_hz <= 0 else 1.0 / tx_rate_hz
        self._rx_publish_period = 0.0 if rx_publish_rate_hz <= 0 else 1.0 / rx_publish_rate_hz

        self.ser = serial.Serial(port=self.port, baudrate=baudrate, timeout=timeout)
        self._rx_buffer = b""

        self._pending_height_cmd: Optional[float] = None
        self._pending_arm_cmd: Optional[tuple[float, float]] = None
        self._pending_pinch_cmd: Optional[tuple[float, float]] = None
        self._last_tx_time: Optional[float] = None

        self._latest_raw_line: Optional[str] = None
        self._latest_raw_rx_time: Optional[float] = None
        self._last_raw_publish_time: Optional[float] = None
        self._last_raw_debug_time: Optional[float] = None

        self._latest_parsed_msg: Optional[ElevatorHeightReading] = None
        self._latest_parsed_rx_time: Optional[float] = None
        self._last_parsed_publish_time: Optional[float] = None
        self._last_parsed_debug_time: Optional[float] = None

        if self.ser.is_open:
            print(
                f"Elevator serial connected: {self.port} @ {baudrate} "
                f"(tx_rate_hz={tx_rate_hz}, rx_publish_rate_hz={rx_publish_rate_hz})"
            )

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "ElevatorSerialConnect":
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
        has_pending = (
            self._pending_height_cmd is not None
            or self._pending_arm_cmd is not None
            or self._pending_pinch_cmd is not None
        )
        return has_pending and (
            self._tx_period <= 0.0
            or self._last_tx_time is None
            or (now - self._last_tx_time) >= self._tx_period
        )

    def _publish_due(self, now: float, last_publish_time: Optional[float]) -> bool:
        return (
            self._rx_publish_period <= 0.0
            or last_publish_time is None
            or (now - last_publish_time) >= self._rx_publish_period
        )

    def _write_pending_if_due(self, force: bool = False) -> bool:
        now = time.monotonic()
        if (
            self._pending_height_cmd is None
            and self._pending_arm_cmd is None
            and self._pending_pinch_cmd is None
        ):
            return False
        if not force and not self._tx_due(now):
            return False

        wrote_any = False
        if self._pending_height_cmd is not None:
            payload = serialize_elevator_cmd(self._pending_height_cmd)
            self.ser.write(payload.encode("ascii"))
            self._debug_dt("TX ELV_CMD", now, self._last_tx_time)
            if self.debug:
                print(f"[TX ELV_CMD] {payload.strip()}")
            self._pending_height_cmd = None
            wrote_any = True

        if self._pending_arm_cmd is not None:
            payload = serialize_arm_cmd(*self._pending_arm_cmd)
            self.ser.write(payload.encode("ascii"))
            if self.debug:
                print(f"[TX ARM_CMD] {payload.strip()}")
            self._pending_arm_cmd = None
            wrote_any = True

        if self._pending_pinch_cmd is not None:
            payload = serialize_pinch_cmd(*self._pending_pinch_cmd)
            self.ser.write(payload.encode("ascii"))
            if self.debug:
                print(f"[TX PINCH_CMD] {payload.strip()}")
            self._pending_pinch_cmd = None
            wrote_any = True

        if wrote_any:
            self.ser.flush()
            self._last_tx_time = now
        return wrote_any

    def send_height_cmd(self, desired_height_m: float, force: bool = False) -> bool:
        self._pending_height_cmd = float(desired_height_m)
        return self._write_pending_if_due(force=force)

    def send_arm_cmd(self, x_m: float, y_m: float, force: bool = False) -> bool:
        self._pending_arm_cmd = (float(x_m), float(y_m))
        return self._write_pending_if_due(force=force)

    def send_pinch_cmd(self, theta1_rad: float, theta2_rad: float, force: bool = False) -> bool:
        self._pending_pinch_cmd = (float(theta1_rad), float(theta2_rad))
        return self._write_pending_if_due(force=force)

    def flush_tx(self, force: bool = False) -> bool:
        return self._write_pending_if_due(force=force)

    def poll(self, max_lines: int = 256) -> int:
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

            parsed = parse_elevator_line(line)
            if parsed is not None:
                self._latest_parsed_msg = parsed
                self._latest_parsed_rx_time = now

        return consumed

    def read(self, max_lines: int = 256) -> List[str]:
        self.poll(max_lines=max_lines)
        if self._latest_raw_line is None or self._latest_raw_rx_time is None:
            return []

        now = time.monotonic()
        if not self._publish_due(now, self._last_raw_publish_time):
            return []

        line = self._latest_raw_line
        self._last_raw_publish_time = now
        self._latest_raw_line = None
        self._debug_dt("RX ELV RAW", now, self._last_raw_debug_time)
        if self.debug:
            age = now - self._latest_raw_rx_time
            print(f"[RX ELV RAW] age={age:.6f}s line={line}")
        self._last_raw_debug_time = now
        self._latest_raw_rx_time = None
        return [line]

    def read_parsed(self, max_lines: int = 256) -> List[ElevatorHeightReading]:
        self.poll(max_lines=max_lines)
        if self._latest_parsed_msg is None or self._latest_parsed_rx_time is None:
            return []

        now = time.monotonic()
        if not self._publish_due(now, self._last_parsed_publish_time):
            return []

        msg = self._latest_parsed_msg
        self._last_parsed_publish_time = now
        self._latest_parsed_msg = None
        self._debug_dt("RX ELV", now, self._last_parsed_debug_time)
        if self.debug:
            age = now - self._latest_parsed_rx_time
            print(f"[RX ELV] age={age:.6f}s msg={msg}")
        self._last_parsed_debug_time = now
        self._latest_parsed_rx_time = None
        return [msg]
