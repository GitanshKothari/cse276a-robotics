
# Seconds between commands. The motor node stops the rover if this
# gaps for more than 0.15 seconds.
COMMAND_DT = 0.05

# Seconds to sit still before each turn.
STOP_PAUSE = 0.50

# Straight line. FORWARD_SPEED is meters per second at FORWARD_CMD.
FORWARD_CMD = 0.30
FORWARD_SPEED = 0.381

# Turning. The motors do not reach full speed instantly, so a turn sweeps
# RATE * (seconds - LAG), not RATE * seconds. Scaling the whole time by the
# angle therefore undershoots small turns: it shrinks the lag along with
# the sweep, and the lag does not shrink.
#
# Measured by direct observation of the rover against floor marks. The IMU
# under-reports rotation badly and must not be used for this.
# Left was fit to three turns (70, 90 and 170 degrees) and matches all of
# them exactly. Right was fit to its 90 degree turn.
LEFT_CMD = 0.40
LEFT_TURN_LAG = 0.238       # seconds lost before the rover is up to speed
LEFT_TURN_RATE = 178.6      # degrees per second once it is

RIGHT_CMD = 0.45
RIGHT_TURN_LAG = 0.238
RIGHT_TURN_RATE = 292.8

# Seconds for a 90 degree turn, derived so there is one source of truth.
# kinematics.py reads these to map a twist onto wheel commands.
LEFT_90_S = LEFT_TURN_LAG + 90.0 / LEFT_TURN_RATE
RIGHT_90_S = RIGHT_TURN_LAG + 90.0 / RIGHT_TURN_RATE
