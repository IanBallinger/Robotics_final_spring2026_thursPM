import serial

def test_serial_con():
    ser = serial.Serial(port='/dev/ttyTHS0', baudrate=9600, timeout=1)
    if ser.is_open:
        print(f"connected to {ser.port}")

    line = ser.readline().decode('utf-8').rstrip()
    print(line)


if __name__=="__main__":
    test_serial_con()
