import time
from collections import deque

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation


PORT = "/dev/tty.usbmodem2101"
BAUD = 115200
CMD_DT = 0.05
WINDOW_SEC = 10.0
MAX_SAMPLES = 5000
NUM_WHEELS = 4
CMD = [6.0, -6.0, 6.0, -6.0]
PLOT = True


class WheelPIDPlotter:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.01)
        time.sleep(2.0)

        self.t0 = time.time()
        self.last_cmd_time = 0.0

        self.t_cmd = deque(maxlen=MAX_SAMPLES)
        self.des_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

        self.t_ack = deque(maxlen=MAX_SAMPLES)
        self.ack_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

        self.t_enc = deque(maxlen=MAX_SAMPLES)
        self.enc_hist = [deque(maxlen=MAX_SAMPLES) for _ in range(NUM_WHEELS)]

    def now(self) -> float:
        return time.time() - self.t0

    def send_command(self):
        now = time.time()
        if now - self.last_cmd_time < CMD_DT:
            return

        line = f"WHL_CMD,{CMD[0]},{CMD[1]},{CMD[2]},{CMD[3]}\n"
        print(line)
        self.ser.write(line.encode("utf-8"))
        self.ser.flush()
        self.last_cmd_time = now

        t = self.now()
        self.t_cmd.append(t)
        for i in range(NUM_WHEELS):
            self.des_hist[i].append(CMD[i])

    def read_serial(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            tag = parts[0]
            t = self.now()

            if tag == "ACK" and len(parts) == 5:
                vals = list(map(float, parts[1:5]))
                self.t_ack.append(t)
                for i in range(NUM_WHEELS):
                    self.ack_hist[i].append(vals[i])
            elif tag == "ENC" and len(parts) == 5:
                print(line)
                vals = list(map(float, parts[1:5]))
                self.t_enc.append(t)
                for i in range(NUM_WHEELS):
                    self.enc_hist[i].append(vals[i])
            else:
                print(line)

    def stop(self):
        try:
            zero_line = "WHL_CMD,0,0,0,0\n"
            for _ in range(5):
                self.ser.write(zero_line.encode("utf-8"))
                self.ser.flush()
                time.sleep(0.02)
            self.ser.write(b"ZERO\n")
            self.ser.flush()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def windowed(self, t_data, y_data):
        if not t_data:
            return [], []

        t_now = self.now()
        t_min = max(0.0, t_now - WINDOW_SEC)
        t_list = list(t_data)
        y_list = list(y_data)

        start = 0
        while start < len(t_list) and t_list[start] < t_min:
            start += 1

        return t_list[start:], y_list[start:]

    def run_plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        desired_lines = []
        ack_lines = []
        enc_lines = []

        for i, ax in enumerate(axes):
            (desired_line,) = ax.plot([], [], label="desired")
            (ack_line,) = ax.plot([], [], "--", label="ack")
            (enc_line,) = ax.plot([], [], label="encoder")
            desired_lines.append(desired_line)
            ack_lines.append(ack_line)
            enc_lines.append(enc_line)

            ax.set_title(f"Wheel {i + 1}")
            ax.set_ylabel("rad/s")
            ax.grid(True)
            ax.legend(loc="upper right")

        axes[2].set_xlabel("time [s]")
        axes[3].set_xlabel("time [s]")

        def update(_frame):
            self.send_command()
            self.read_serial()

            t_now = self.now()
            t_min = max(0.0, t_now - WINDOW_SEC)

            for i, ax in enumerate(axes):
                td, yd = self.windowed(self.t_cmd, self.des_hist[i])
                ta, ya = self.windowed(self.t_ack, self.ack_hist[i])
                te, ye = self.windowed(self.t_enc, self.enc_hist[i])

                desired_lines[i].set_data(td, yd)
                ack_lines[i].set_data(ta, ya)
                enc_lines[i].set_data(te, ye)

                ax.set_xlim(t_min, t_min + WINDOW_SEC)

                y_all = yd + ya + ye if (yd or ya or ye) else [0.0, CMD[i]]
                ymin = min(y_all)
                ymax = max(y_all)
                pad = max(0.5, 0.1 * max(abs(ymin), abs(ymax), 1.0))
                ax.set_ylim(ymin - pad, ymax + pad)

            return (*desired_lines, *ack_lines, *enc_lines)

        self.ani = FuncAnimation(
            fig, update, interval=50, blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
        self.stop()

    def run_no_plot(self):
        try:
            while True:
                self.send_command()
                self.read_serial()
                time.sleep(0.01)
        finally:
            self.stop()


def main():
    print(f"Using port {PORT} @ {BAUD}")
    print(f"Sending WHL_CMD,{CMD[0]},{CMD[1]},{CMD[2]},{CMD[3]}")
    plotter = WheelPIDPlotter(PORT, BAUD)
    try:
        if PLOT:
            plotter.run_plot()
        else:
            plotter.run_no_plot()
    except KeyboardInterrupt:
        plotter.stop()


if __name__ == "__main__":
    main()
