import time
import serial
from serialization import serialize_whl

class SerialConnect():
    def __init__(self):
        self.ser = serial.Serial(port='/dev/ttyESP', baudrate=115200, timeout=1)
        if self.ser:
            print("Connected!")

    def read(self):
        pass

    def send(self, data):
        data_str = serialize_whl(0,0,0,0)
        print(data_str.encode('utf-8'))
        self.ser.write(data_str.encode('utf-8'))
        self.ser.flush()

if __name__=='__main__':
    ser_con = SerialConnect()
    while True:
        ser_con.send(0)
        time.sleep(0.5)
