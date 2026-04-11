import re

def serialize_whl(lb,rb,lf,rf):
    return f"WHL_CMD,{lb},{rb},{lf},{rf}\n"

def deserialize_imu(string):
    pass
