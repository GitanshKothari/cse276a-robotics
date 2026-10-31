"""Turn a unicycle twist into the [L, R] pair the motor node expects.

The rover takes two normalised wheel commands in [-0.5, 0.5] and has no
encoders, so the mapping comes from the HW1 hand calibration: a known
forward command gives a known speed, and a known differential command
turns 90 degrees in a known time. Left and right spins were timed
separately because the drivetrain is not symmetric.
"""

import math

from robot_control.config import (
    FORWARD_CMD, FORWARD_SPEED,
    LEFT_CMD, LEFT_TURN_RATE,
    RIGHT_CMD, RIGHT_TURN_RATE,
)

# Hard limit enforced by the motor node, which clips to +/-0.5.
CMD_LIMIT = 0.5

# Command units per m/s of body velocity.
LINEAR_GAIN = FORWARD_CMD / FORWARD_SPEED

# Yaw rate produced by a pure differential command, per direction.
#
# These are the steady rates, not 90 degrees divided by LEFT_90_S. That
# division would fold in the startup lag, because a timed turn starts from
# rest and loses 0.24 s before the wheels are up to speed. The open-loop
# planner has to pay that lag. This controller does not: it commands every
# tick at 20 Hz and never stops, so it sees the steady rate. Using the
# timed average here made the gains 47% and 77% too high, which told the
# controller it was turning far slower than it was, and made it overshoot
# every heading.

LEFT_RATE = math.radians(LEFT_TURN_RATE)
RIGHT_RATE = math.radians(RIGHT_TURN_RATE)

# Command units per rad/s, again per direction.
LEFT_GAIN = LEFT_CMD / LEFT_RATE
RIGHT_GAIN = RIGHT_CMD / RIGHT_RATE

# Fastest twist the calibration actually covers. Going past this means
# extrapolating the linear fit into the part of the curve where the motors
# saturate, so the controller clamps here instead.
MAX_LINEAR = CMD_LIMIT / LINEAR_GAIN
MAX_ANGULAR_LEFT = CMD_LIMIT / LEFT_GAIN
MAX_ANGULAR_RIGHT = CMD_LIMIT / RIGHT_GAIN


def angular_gain(omega):
    """Command units per rad/s for the direction omega turns in."""
    return LEFT_GAIN if omega >= 0.0 else RIGHT_GAIN


def twist_to_wheels(v, omega, min_cmd=0.0, min_turn_cmd=None):
    """Wheel commands for a body twist.

    Saturation scales both wheels together so the ratio of turning to
    driving survives - clipping them independently would silently bend the
    path. ``min_cmd`` lifts a command that is too small to overcome static
    friction up to the point where the wheels actually break away.

    A turn needs a bigger push than a drive. Rolling forward is mostly
    rolling resistance, but a spin on the spot drags both tyres sideways
    across the floor, and dry rubber scrubbing is far harder to start.
    Measured on this rover, 0.10 rolls it forward while a spin needs about
    0.40. One floor cannot serve both: at 0.10 the rover drove its straight
    legs and simply refused to turn.

    So ``min_turn_cmd`` gives turning its own floor, and the two are
    blended by how much of this command is differential. A pure drive gets
    ``min_cmd``, a pure spin gets ``min_turn_cmd``, and an arc gets a
    value in between.
    """
    linear = LINEAR_GAIN * v
    differential = angular_gain(omega) * omega

    left = linear - differential
    right = linear + differential

    peak = max(abs(left), abs(right))
    if peak < 1e-9:
        return 0.0, 0.0

    turn_fraction = min(1.0, abs(differential) / peak)

    if peak > CMD_LIMIT:
        scale = CMD_LIMIT / peak
        left *= scale
        right *= scale
        peak = CMD_LIMIT

    floor = min_cmd
    if min_turn_cmd is not None:
        floor = min_cmd + (min_turn_cmd - min_cmd) * turn_fraction
    floor = min(floor, CMD_LIMIT)

    if 0.0 < peak < floor:
        scale = floor / peak
        left *= scale
        right *= scale

    return left, right


def wheels_to_twist(left, right):
    """Inverse of :func:`twist_to_wheels`, used to predict from commands.

    The filter propagates on the twist the controller asked for, so it has
    to read the same calibration back out of the wheel pair it published.
    """
    linear = 0.5 * (left + right)
    differential = 0.5 * (right - left)
    v = linear / LINEAR_GAIN
    omega = differential / angular_gain(differential)
    return v, omega
