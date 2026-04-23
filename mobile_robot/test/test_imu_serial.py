import serial
import time


SERIAL_PORT = "/dev/tty.usbmodem1101"
BAUD_RATE = 115200
TIMEOUT = 1.0

PRINT_EVERY_N = 10   # print 1 out of every 10 valid IMU readings


def parse_imu_line(line: str):
    """
    Expected line format:
    time_us\tax\tay\taz\twx\twy\twz
    """
    parts = line.strip().split()

    if len(parts) != 7:
        return None

    try:
        data = {
            "time_us": int(parts[0]),
            "ax": round(float(parts[1]), 2),
            "ay": round(float(parts[2]), 2),
            "az": round(float(parts[3]), 2),
            "wx": round(float(parts[4]), 2),
            "wy": round(float(parts[5]), 2),
            "wz": round(float(parts[6]), 2),
        }
        return data
    except ValueError:
        return None


def main():
    print(f"Opening serial port: {SERIAL_PORT} @ {BAUD_RATE}")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

    time.sleep(2)
    print("Listening for IMU data...\n")

    valid_count = 0

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            data = parse_imu_line(line)
            if data is None:
                print(f"[INFO] {line}")
                continue

            valid_count += 1

            # discard intermediate readings for printing purposes
            if valid_count % PRINT_EVERY_N != 0:
                continue

            print(
                f"t={data['time_us']:>10} "
                f"ax={data['ax']:+7.2f} ay={data['ay']:+7.2f} az={data['az']:+7.2f} "
                f"wx={data['wx']:+7.2f} wy={data['wy']:+7.2f} wz={data['wz']:+7.2f}"
            )

    except KeyboardInterrupt:
        print("\nStopping serial reader.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()