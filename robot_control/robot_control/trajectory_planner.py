import math

from .config import (
    STOP_PAUSE,
    FORWARD_CMD, FORWARD_SPEED,
    LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S,
)


# Read x, y, heading. Blank lines and # comments are skipped.
def load_waypoints(path):
    points = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y, heading = line.split(",")
            points.append((float(x), float(y), float(heading)))
    return points


# Keep a turn between -180 and 180 degrees.
# At (2,2), facing +y, the turn toward (1,1) is -225 degrees.
# This folds it to +135 degrees, so the rover turns left.
def shortest_turn(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# A turn is already between -180 and 180, so split at most once.
# 135 degrees becomes 90, then 45.
def split_long_turn(angle_deg):
    abs_angle = abs(angle_deg)
    if abs_angle == 0.0:
        return []

    angle_sign = 1.0 if angle_deg > 0.0 else -1.0
    if abs_angle > 90.0:
        return [angle_sign * 90.0, angle_sign * (abs_angle - 90.0)]
    return [angle_sign * abs_angle]


# add a stop command and then a turn command. speed and time is determined by the angle and direction (left vs right).
def add_turn(moves, angle_deg):
    if angle_deg > 0.0:
        left = -LEFT_CMD
        right = LEFT_CMD
        seconds = LEFT_90_S * (angle_deg / 90.0)
    else:
        left = RIGHT_CMD
        right = -RIGHT_CMD
        seconds = RIGHT_90_S * (abs(angle_deg) / 90.0)
    moves.append((STOP_PAUSE, 0.0, 0.0))
    moves.append((seconds, left, right))


# Add one forward command.
def add_forward(moves, distance):
    seconds = distance / FORWARD_SPEED
    moves.append((seconds, FORWARD_CMD, FORWARD_CMD))



def make_plan(points):
    moves = []
    current_angle = points[0][2]

    for here, nxt in zip(points, points[1:]):
        dx = nxt[0] - here[0]
        dy = nxt[1] - here[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            continue

        target_angle = math.atan2(dy, dx)
        turn_angles = split_long_turn(math.degrees(shortest_turn(target_angle - current_angle)))
        for i in (turn_angles):
            add_turn(moves, i)

        add_forward(moves, distance)
        current_angle = target_angle

    moves.append((STOP_PAUSE, 0.0, 0.0))
    return moves
