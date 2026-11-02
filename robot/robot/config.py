"""All tunable numbers. Change here, nowhere else."""

import math

# -------- motor calibration --------
# Wheel command and the real forward speed at that command.
FORWARD_CMD = 0.20
FORWARD_SPEED = 0.195         # metres per second
LEFT_CMD = 0.35
LEFT_90_S = 0.85              # seconds to pivot 90 degrees LEFT
RIGHT_CMD = 0.30
RIGHT_90_S = 1.00             # seconds to pivot 90 degrees RIGHT

# -------- motion profile --------
STEP_DISTANCE = 0.25          # metres per forward step
STOP_PAUSE = 0.20             # seconds the robot sits still after a step
COMMAND_DT = 0.05             # 20 Hz state-machine tick
GLIDE_CMD = 0.08              # slow continuous forward during localization
RAMP_SECONDS = 0.30           # trapezoid ramp at the start and end of a move

# -------- waypoint tolerances --------
ALONG_TOLERANCE = 0.10        # within this distance a waypoint is "reached"
LATERAL_TOLERANCE = 0.15      # cross-track margin (axis-margin mode only)
HEADING_TOLERANCE = math.radians(15)
MAX_CORRECTION = math.radians(30)   # cap a heading-correction turn

# -------- pose filter --------
MAX_POSE_JUMP = 0.5           # metres, outlier threshold vs last good pose
MAX_HEADING_JUMP = math.radians(45)
FILTER_RESET_TIMEOUT = 2.0    # seconds of rejections before the baseline resets

# -------- tag localization --------
MAX_TAG_AGE = 1.0             # drop detections older than this
MAX_WAIT_FOR_TAG = 20.0       # LOCALIZE timeout -> stop
CAMERA_FRAME = "camera_frame"
TAG_FRAME_PREFIX = "tag36h11:"

# -------- serial / motor node --------
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
MOTOR_SAFETY_TIMEOUT = 0.15   # motor stops if no command for this long
MOTOR_TX_HZ = 20
IMU_TX_HZ = 50

# -------- behaviour flags --------
DRY_RUN = False               # True = log but do not drive wheels
USE_AXIS_MARGIN = False       # True = separate along/cross tolerances

# -------- log files --------
SPEED_LOG = "/tmp/speed_log.csv"
POSE_LOG = "/tmp/pose_log.csv"
