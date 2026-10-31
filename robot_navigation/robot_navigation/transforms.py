"""Small SE(3) helpers.

Everything here is plain numpy so the package has no tf/transforms3d
dependency at runtime. A "transform" is always a 4x4 homogeneous matrix.
"""

import math

import numpy as np

# Rotation from a base-style frame (x forward, y left, z up) to an optical
# frame (x right, y down, z forward). OpenCV/solvePnP and therefore the
# AprilTag poses live in the optical frame.
BASE_TO_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def wrap_angle(angle):
    """Fold an angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(rotation, translation):
    """Pack a 3x3 rotation and a 3-vector into a 4x4 transform."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def invert_transform(transform):
    """Inverse of a rigid transform, without a general matrix inverse."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def yaw_of(transform):
    """Rotation about world z, i.e. where the frame's x axis points."""
    return math.atan2(transform[1, 0], transform[0, 0])


def planar_pose(transform):
    """Reduce a 4x4 transform to the (x, y, theta) the filter works in."""
    return float(transform[0, 3]), float(transform[1, 3]), yaw_of(transform)


def quaternion_from_yaw(yaw):
    """(x, y, z, w) for a rotation about z."""
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def yaw_from_quaternion(x, y, z, w):
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def camera_extrinsics(x, y, z, mount_yaw, mount_pitch, mount_roll=0.0):
    """Transform from base_link to the camera optical frame.

    The mount angles describe how the camera body is bolted to the rover:
    positive ``mount_yaw`` turns it to the left, positive ``mount_pitch``
    tips it downwards (the usual way a rover camera is aimed at the floor).
    """
    rotation = rot_z(mount_yaw) @ rot_y(mount_pitch) @ rot_x(mount_roll) @ BASE_TO_OPTICAL
    return make_transform(rotation, (x, y, z))


def tag_extrinsics(x, y, z, normal_yaw, tilt=0.0):
    """World pose of a tag given where its face points.

    A tag is described by the position of its centre and ``normal_yaw``, the
    world direction its printed face looks towards. ``tilt`` leans the face
    up (positive) or down about its own horizontal axis, for tags that are
    not mounted perfectly vertically.

    In the AprilTag convention the tag's own x axis runs to the right across
    the face, y runs up it, and z sticks out of it towards the viewer.
    """
    z_axis = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
    x_axis = np.array([-math.sin(normal_yaw), math.cos(normal_yaw), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if tilt:
        # Lean about the tag's own x axis, which is horizontal by construction.
        rotation = rotation @ rot_x(tilt)
    return make_transform(rotation, (x, y, z))
