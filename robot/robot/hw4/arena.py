BOUNDS = (0.0, 0.0, 2.97, 3.00)
START = (0.70, 2.30, 0.0)              # axle x, y, heading (+x = 0, CCW positive)
GOAL = (2.31, 0.64)
CYLINDER = (1.45, 1.55, 0.31)          # centre x, centre y, radius
ROBOT_RADIUS = 0.07                    # axle to the furthest part of the robot
PLANNING_MARGIN = 0.10
CAMERA_OFFSET = (0.11, 0.0)            # camera forward/left of the axle
CAMERA_YAW = 0.059
CAMERA_FRAME = "camera_frame"

BOUNDARY_TAG_FAMILY = "36h11"
TAG_SIZE = 0.165
BOUNDARY_TAGS = {
    0: (2.97, 0.95), 7: (2.97, 1.98),
    1: (1.98, 3.00), 3: (0.99, 3.00),
    4: (0.00, 1.98), 8: (0.00, 1.00),
    9: (0.98, 0.00), 5: (1.97, 0.00),
}

OBSTACLE_TAG_FAMILY = "Standard41h12"
OBSTACLE_TAGS = (1, 2, 3)
OBSTACLE_TAG_SIZE = 0.19 * 5 / 9       # detection square is 5 of the 9 cells
TAG_STANDOFF = 0.0             

SURVEY_STOPS = ((0.70, 2.30), (0.45, 1.80), (0.45, 1.25), (0.70, 0.70), (1.485, 0.70),
                (2.27, 0.70), (2.55, 1.25), (2.55, 1.80), (2.27, 2.30),
                (1.75, 2.60), (1.20, 2.60))

FORWARD_CMD, FORWARD_SPEED = 0.20, 0.230
LEFT_CMD, LEFT_90_S = 0.35, 0.85
RIGHT_CMD, RIGHT_90_S = 0.30, 1.00
STEP_M = 0.12
