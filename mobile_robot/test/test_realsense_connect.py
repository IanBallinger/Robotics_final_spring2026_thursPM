import pyrealsense2 as rs    

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
                                                                                                                                                    
pipeline = rs.pipeline()                                                                                                                             
config = rs.config()                                                                                                                                 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)                                                                                  
                                                                                                                                                    
profile = pipeline.start(config)                                                                                                                     
print("started ok")                                                                                                                                  
frames = pipeline.wait_for_frames()                                                                                                                  
print("got frames")                                                                                                                                  
pipeline.stop()