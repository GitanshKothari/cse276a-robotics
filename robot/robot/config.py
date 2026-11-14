import math


FORWARD_CMD = 0.20
FORWARD_SPEED = 0.230         # metres per second (last measured forward cal)
LEFT_CMD = 0.35
LEFT_90_S = 0.82            # seconds to pivot 90 degrees LEFT (turn test: 4 turns gave 410 deg at 0.93)
RIGHT_CMD = 0.30
RIGHT_90_S = 1.00             # seconds to pivot 90 degrees RIGHT


STEP_DISTANCE = 0.25         
STOP_PAUSE = 0.20            
COMMAND_DT = 0.05            
GLIDE_CMD = 0.08             
RAMP_SECONDS = 0.30          


ALONG_TOLERANCE = 0.10        
LATERAL_TOLERANCE = 0.15      
HEADING_TOLERANCE = math.radians(15)
MAX_CORRECTION = math.radians(30)   


MAX_POSE_JUMP = 0.5           
MAX_HEADING_JUMP = math.radians(45)
FILTER_RESET_TIMEOUT = 2.0    


MAX_TAG_AGE = 1.0             
MAX_WAIT_FOR_TAG = 20.0      
CAMERA_FRAME = "camera_frame"
TAG_FRAME_PREFIX = "tag36h11:"


SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
MOTOR_SAFETY_TIMEOUT = 0.15   
MOTOR_TX_HZ = 20
IMU_TX_HZ = 50


SPEED_LOG = "/tmp/speed_log.csv"
POSE_LOG = "/tmp/pose_log.csv"
