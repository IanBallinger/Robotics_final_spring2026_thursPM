import numpy as np


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, error: float):
        return self.kp * error + self.ki * error.integral + self.kd * error.derivative


class MobileRobotController(PIDController):
    def __init__(self, kp: float, ki: float, kd: float):
        super().__init__(kp, ki, kd)


class ElevatorController(PIDController):
    def __init__(self, kp: float, ki: float, kd: float):
        super().__init__(kp, ki, kd)
