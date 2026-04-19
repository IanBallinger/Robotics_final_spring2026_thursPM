import time
import threading
from collections import deque

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation


PORT = "/dev/ttyESP"
BAUD = 115200

CMD_DT = 0.05          # seconds between command transmissions
WINDOW_SEC = 10.0      # plot window length
MAX_SAMPLES = 5000
NUM_WHEELS = 4


class LiveWheelPIDTuner:
    """
    Host-side tool for mobile_robot/src/esp32/src/wheels/wheel_pid_tuner.cpp.

    Supported host -> MCU commands:
      WHL_CMD,w1,w2,w3,w4
      PID_ALL,kp,ki,kd
      PID_WHEEL,i,kp,ki,kd
      ZERO
      GET_PID

    Expected MCU -> host telemetry:
      ACK,w1,w2,w3,w4
      ENC,v1,v2,v3,v4
      EFF,u1,u2,u3,u4
      PID,kp1,ki1,kd1,...,kp4,ki4,kd4
      STATUS,...
      ERR,...
    """

    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.05)
        time.sleep(2.0)

        self.running = True
        self.t0 = time.time()
        self.lock = threading.Lock()

        self.desired = [0.0] * NUM_WHEELS
        self.latest_pid = [(0.0, 0.0, 0.0) for _ in range(NUM_WHEELS)]

        self.t_cmd = deque(maxlen=MAX_SAMPLES)
        self.des_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

        self.t_ack = deque(maxlen=MAX_SAMPLES)
        self.ack_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

        self.t_enc = deque(maxlen=MAX_SAMPLES)
        self.enc_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

        self.t_eff = deque(maxlen=MAX_SAMPLES)
        self.eff_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

    def now(self) -> float:
        return time.time() - self.t0

    def make_wheel_cmd(self, w1: float, w2: float, w3: float, w4: float) -> str:
        return f"WHL_CMD,{w1},{w2},{w3},{w4}\n"

    def make_pid_all_cmd(self, kp: float, ki: float, kd: float) -> str:
        return f"PID_ALL,{kp},{ki},{kd}\n"

    def make_pid_wheel_cmd(self, wheel_idx: int, kp: float, ki: float, kd: float) -> str:
        return f"PID_WHEEL,{wheel_idx},{kp},{ki},{kd}\n"

    def send_line(self, line: str):
        self.ser.write(line.encode("utf-8"))
        self.ser.flush()

    def send_loop(self):
        next_send = time.time()
        while self.running:
            now = time.time()
            if now >= next_send:
                with self.lock:
                    cmd = list(self.desired)

                self.send_line(self.make_wheel_cmd(*cmd))

                t = self.now()
                self.t_cmd.append(t)
                for i in range(NUM_WHEELS):
                    self.des_hist[i].append(cmd[i])

                next_send += CMD_DT
            else:
                time.sleep(0.001)

    def read_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                t = self.now()
                parts = line.split(",")
                tag = parts[0]

                if tag == "ACK" and len(parts) == 5:
                    vals = list(map(float, parts[1:5]))
                    self.t_ack.append(t)
                    for i in range(NUM_WHEELS):
                        self.ack_hist[i].append(vals[i])

                elif tag == "ENC" and len(parts) == 5:
                    vals = list(map(float, parts[1:5]))
                    self.t_enc.append(t)
                    for i in range(NUM_WHEELS):
                        self.enc_hist[i].append(vals[i])

                elif tag == "EFF" and len(parts) == 5:
                    vals = list(map(float, parts[1:5]))
                    self.t_eff.append(t)
                    for i in range(NUM_WHEELS):
                        self.eff_hist[i].append(vals[i])

                elif tag == "PID" and len(parts) == 13:
                    gains = list(map(float, parts[1:13]))
                    self.latest_pid = [tuple(gains[3 * i: 3 * i + 3]) for i in range(NUM_WHEELS)]
                    print("PID gains:")
                    for i, (kp, ki, kd) in enumerate(self.latest_pid, start=1):
                        print(f"  wheel {i}: Kp={kp}, Ki={ki}, Kd={kd}")

                elif tag == "STATUS":
                    print("STATUS:", ",".join(parts[1:]))

                elif tag == "ERR":
                    print("ESP32 error:", ",".join(parts[1:]))

                else:
                    print("RX:", line)

            except Exception as e:
                print("read error:", e)

    def input_loop(self):
        print("4-wheel PID tuner")
        print("Commands:")
        print("  lr <left> <right>            -> sets [left, right, left, right]")
        print("  cmd <w1> <w2> <w3> <w4>     -> sets explicit wheel commands")
        print("  pid <kp> <ki> <kd>          -> applies same gains to all wheels")
        print("  pid <wheel> <kp> <ki> <kd>  -> applies gains to one wheel (1..4)")
        print("  zero                        -> send ZERO and set desired command to zero")
        print("  getpid                      -> request current gains")
        print("  q                           -> quit\n")

        while self.running:
            try:
                s = input("cmd> ").strip()
                if not s:
                    continue

                if s.lower() in {"q", "quit", "exit"}:
                    self.running = False
                    break

                parts = s.split()
                cmd = parts[0].lower()

                if cmd == "lr" and len(parts) == 3:
                    left = float(parts[1])
                    right = float(parts[2])
                    vals = [left, right, left, right]
                    with self.lock:
                        self.desired = vals
                    print(f"Desired = {vals}")

                elif cmd == "cmd" and len(parts) == 5:
                    vals = list(map(float, parts[1:5]))
                    with self.lock:
                        self.desired = vals
                    print(f"Desired = {vals}")

                elif cmd == "pid" and len(parts) == 4:
                    kp, ki, kd = map(float, parts[1:4])
                    self.send_line(self.make_pid_all_cmd(kp, ki, kd))
                    print(f"Sent PID_ALL: Kp={kp}, Ki={ki}, Kd={kd}")

                elif cmd == "pid" and len(parts) == 5:
                    wheel = int(parts[1])
                    kp, ki, kd = map(float, parts[2:5])
                    if wheel not in {1, 2, 3, 4}:
                        print("wheel must be 1..4")
                        continue
                    self.send_line(self.make_pid_wheel_cmd(wheel, kp, ki, kd))
                    print(f"Sent PID_WHEEL {wheel}: Kp={kp}, Ki={ki}, Kd={kd}")

                elif cmd == "zero" and len(parts) == 1:
                    with self.lock:
                        self.desired = [0.0] * NUM_WHEELS
                    self.send_line("ZERO\n")
                    print("Zeroed wheel commands")

                elif cmd == "getpid" and len(parts) == 1:
                    self.send_line("GET_PID\n")

                else:
                    print("Unknown command. Use lr/cmd/pid/zero/getpid/q")

            except Exception as e:
                print("input error:", e)

    def start_threads(self):
        self.tx_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.rx_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.in_thread = threading.Thread(target=self.input_loop, daemon=True)

        self.tx_thread.start()
        self.rx_thread.start()
        self.in_thread.start()

        self.send_line("GET_PID\n")

    def stop(self):
        self.running = False
        try:
            self.send_line("ZERO\n")
            for _ in range(5):
                self.send_line(self.make_wheel_cmd(0.0, 0.0, 0.0, 0.0))
                time.sleep(0.02)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def _windowed(self, t_data, y_data):
        if not t_data:
            return [], []
        t_now = self.now()
        t_min = max(0.0, t_now - WINDOW_SEC)
        idx = 0
        for i, t in enumerate(t_data):
            if t >= t_min:
                idx = i
                break
        return list(t_data)[idx:], list(y_data)[idx:]

    def run_plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        desired_lines = []
        ack_lines = []
        actual_lines = []

        for i, ax in enumerate(axes):
            line_des, = ax.plot([], [], label=f"desired w{i + 1}", linewidth=2)
            line_ack, = ax.plot([], [], label=f"ACK w{i + 1}", linewidth=1.5, linestyle="--")
            line_act, = ax.plot([], [], label=f"actual w{i + 1}", linewidth=2)
            desired_lines.append(line_des)
            ack_lines.append(line_ack)
            actual_lines.append(line_act)
            ax.set_title(f"Wheel {i + 1}")
            ax.set_ylabel("velocity [rad/s]")
            ax.grid(True)
            ax.legend(loc="upper right")

        axes[2].set_xlabel("time [s]")
        axes[3].set_xlabel("time [s]")
        fig.suptitle("4-wheel PID tuning: desired vs ACK vs measured encoder velocity")

        def update(_frame):
            t_now = self.now()
            t_min = max(0.0, t_now - WINDOW_SEC)

            for i, ax in enumerate(axes):
                td, yd = self._windowed(self.t_cmd, self.des_hist[i])
                ta, ya = self._windowed(self.t_ack, self.ack_hist[i])
                te, ye = self._windowed(self.t_enc, self.enc_hist[i])

                desired_lines[i].set_data(td, yd)
                ack_lines[i].set_data(ta, ya)
                actual_lines[i].set_data(te, ye)

                ax.set_xlim(t_min, t_min + WINDOW_SEC)

                y_all = yd + ya + ye if (yd or ya or ye) else [0.0]
                ymin, ymax = min(y_all), max(y_all)
                pad = max(0.5, 0.1 * max(abs(ymin), abs(ymax), 1.0))
                ax.set_ylim(ymin - pad, ymax + pad)

            return (*desired_lines, *ack_lines, *actual_lines)

        _ani = FuncAnimation(fig, update, interval=100, blit=False)
        plt.tight_layout()
        plt.show()

        self.stop()


if __name__ == "__main__":
    tuner = LiveWheelPIDTuner(PORT, BAUD)
    tuner.start_threads()
    tuner.run_plot()
