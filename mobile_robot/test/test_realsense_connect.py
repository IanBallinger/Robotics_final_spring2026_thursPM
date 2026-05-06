import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"device: {name} serial={serial}")

    for sensor in dev.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        print(f"  sensor: {sensor_name}")
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                vsp = profile.as_video_stream_profile()
                dims = f"{vsp.width()}x{vsp.height()}"
            else:
                dims = "non-video"
            print(
                "    profile:",
                profile.stream_name(),
                dims,
                profile.fps(),
                profile.format(),
            )

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

profile = pipeline.start(config)
print("started ok")
frames = pipeline.wait_for_frames()
print("got frames")
pipeline.stop()
