import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(name, serial)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

profile = pipeline.start(config)
print("started ok")
frames = pipeline.wait_for_frames()
print("got frames")
pipeline.stop()
