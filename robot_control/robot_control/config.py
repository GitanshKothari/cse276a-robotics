
# Seconds between commands. The motor node stops the rover if this
# gaps for more than 0.15 seconds.
COMMAND_DT = 0.05

# Seconds to sit still before each turn.
STOP_PAUSE = 0.50

# Straight line. FORWARD_SPEED is meters per second at FORWARD_CMD.
FORWARD_CMD = 0.30
FORWARD_SPEED = 0.294

# Seconds a 90 degree turn takes. A 45 degree turn takes half of that.
LEFT_CMD = 0.40
LEFT_90_S = 1.19
RIGHT_CMD = 0.45
RIGHT_90_S = 1.62
