import pyrealsense2 as rs    

                                                                                                                                                    
pipeline = rs.pipeline()                                                                                                                             
config = rs.config()                                                                                                                                 
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)                                                                                  
# config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)                                                                                   
                                                                                                                                                    
profile = pipeline.start(config)                                                                                                                     
print("started ok")                                                                                                                                  
frames = pipeline.wait_for_frames()                                                                                                                  
print("got frames")                                                                                                                                  
pipeline.stop()
