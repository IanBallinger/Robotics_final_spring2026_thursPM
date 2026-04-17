import time
import threading
from collections import deque

import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


PORT = "/dev/ttyESP"
BAUD = 115200

CMD_DT = 0.05          # seconds between command transmissions
WINDOW_SEC = 10.0      # plot window length
MAX_SAMPLES = 5000


class LiveWheelPIDTuner:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.05)
        time.sleep(2.0)

        self.running = True
        self.t0 = time.time()

        self.des_left = 0.0
        self.des_right = 0.0
        self.lock = threading.Lock()

        self.t_cmd = deque(maxlen=MAX_SAMPLES)
        self.des_l_hist = deque(maxlen=MAX_SAMPLES)
        self.des_r_hist = deque(maxlen=MAX_SAMPLES)

        self.t_enc_l = deque(maxlen=MAX_SAMPLES)
        self.enc_l = deque(maxlen=MAX_SAMPLES)

        self.t_enc_r = deque(maxlen=MAX_SAMPLES)
        self.enc_r = deque(maxlen=MAX_SAMPLES)

    def now(self) -> float:
        return time.time() - self.t0

    def make_cmd(self, w1: float, w2: float, w3: float = 0.0, w4: float = 0.0) -> str:
        return f"WHL_CMD,{w1},{w2},{w3},{w4}\n"

    def send_loop(self):
        next_send = time.time()
        while self.running:
            now = time.time()
            if now >= next_send:
                with self.lock:
                    wl = self.des_left
                    wr = self.des_right

                msg = self.make_cmd(wl, wr)
                self.ser.write(msg.encode("utf-8"))
                self.ser.flush()

                t = self.now()
                self.t_cmd.append(t)
                self.des_l_hist.append(wl)
                self.des_r_hist.append(wr)

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

                if parts[0] == "ENC_L" and len(parts) == 2:
                    self.t_enc_l.append(t)
                    self.enc_l.append(float(parts[1]))
                elif parts[0] == "ENC_R" and len(parts) == 2:
                    self.t_enc_r.append(t)
                    self.enc_r.append(float(parts[1]))
                elif parts[0] not in ("ACK",):
                    print("RX:", line)

            except Exception as e:
                print("read error:", e)

    def input_loop(self):
        print("Enter desired wheel speeds as: left right")
        print("Examples:")
        print("  5 5")
        print("  8 6")
        print("  0 0")
        print("Type q to quit.\n")

        while self.running:
            try:
                s = input("cmd> ").strip()
                if s.lower() in {"q", "quit", "exit"}:
                    self.running = False
                    break

                parts = s.split()
                if len(parts) != 2:
                    print("Expected: <left_speed> <right_speed>")
                    continue

                wl = float(parts[0])
                wr = float(parts[1])

                with self.lock:
                    self.des_left = wl
                    self.des_right = wr

                print(f"Set desired speeds: left={wl}, right={wr}")

            except Exception as e:
                print("input error:", e)

    def start_threads(self):
        self.tx_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.rx_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.in_thread = threading.Thread(target=self.input_loop, daemon=True)

        self.tx_thread.start()
        self.rx_thread.start()
        self.in_thread.start()

    def stop(self):
        self.running = False
        try:
            for _ in range(5):
                self.ser.write(self.make_cmd(0.0, 0.0).encode("utf-8"))
                self.ser.flush()
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        line_des_l, = ax1.plot([], [], label="desired left")
        line_act_l, = ax1.plot([], [], label="actual left")
        ax1.set_ylabel("left wheel speed")
        ax1.grid(True)
        ax1.legend()

        line_des_r, = ax2.plot([], [], label="desired right")
        line_act_r, = ax2.plot([], [], label="actual right")
        ax2.set_ylabel("right wheel speed")
        ax2.set_xlabel("time [s]")
        ax2.grid(True)
        ax2.legend()

        def update(_frame):
            tdl, ydl = self._windowed(self.t_cmd, self.des_l_hist)
            tdr, ydr = self._windowed(self.t_cmd, self.des_r_hist)
            tel, yel = self._windowed(self.t_enc_l, self.enc_l)
            ter, yer = self._windowed(self.t_enc_r, self.enc_r)

            line_des_l.set_data(tdl, ydl)
            line_act_l.set_data(tel, yel)

            line_des_r.set_data(tdr, ydr)
            line_act_r.set_data(ter, yer)

            t_now = self.now()
            t_min = max(0.0, t_now - WINDOW_SEC)

            ax1.set_xlim(t_min, t_min + WINDOW_SEC)
            ax2.set_xlim(t_min, t_min + WINDOW_SEC)

            y_all_left = ydl + yel if (ydl or yel) else [0.0]
            y_all_right = ydr + yer if (ydr or yer) else [0.0]

            lmin, lmax = min(y_all_left), max(y_all_left)
            rmin, rmax = min(y_all_right), max(y_all_right)

            lpad = max(0.5, 0.1 * max(abs(lmin), abs(lmax), 1.0))
            rpad = max(0.5, 0.1 * max(abs(rmin), abs(rmax), 1.0))

            ax1.set_ylim(lmin - lpad, lmax + lpad)
            ax2.set_ylim(rmin - rpad, rmax + rpad)

            return line_des_l, line_act_l, line_des_r, line_act_r

        ani = FuncAnimation(fig, update, interval=100, blit=False)
        plt.tight_layout()
        plt.show()

        self.stop()


if __name__ == "__main__":
    tuner = LiveWheelPIDTuner(PORT, BAUD)
    tuner.start_threads()
    tuner.run_plot()