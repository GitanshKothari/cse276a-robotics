"""Vision: turn an AprilTag detection into a world pose for the robot.

Two small classes:
- PoseEstimator  — pure maths (tag + camera + world transforms).
- TagLocalizer   — ROS TF lookup, freshness filter, outlier filter.
"""

import math
import time
import numpy as np

from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .config import (
    CAMERA_FRAME, TAG_FRAME_PREFIX,
    MAX_TAG_AGE, MAX_POSE_JUMP, MAX_HEADING_JUMP,
    FILTER_RESET_TIMEOUT,
)


def wrap(angle):
    """Fold any angle into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _transform(position, quaternion):
    """Build a 4x4 pose matrix from a position and a quaternion [x,y,z,w]."""
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


class PoseEstimator:
    """Pure maths. Given a tag seen in the camera, return the robot pose
    in the world frame."""

    def __init__(self, landmarks, camera_mount):
        self.landmarks = landmarks
        # Camera to robot body. Fixed because the camera is bolted on.
        self.T_BC = _transform(
            camera_mount["position"], camera_mount["rotation"]
        )

    def estimate(self, tag_id, position, quaternion):
        tag = self.landmarks[tag_id]
        T_WT = _transform(tag["position"], tag["rotation"])
        T_CT = _transform(position, quaternion)
        T_WB = T_WT @ np.linalg.inv(T_CT) @ np.linalg.inv(self.T_BC)
        x = T_WB[0, 3]
        y = T_WB[1, 3]
        theta = math.atan2(T_WB[1, 0], T_WB[0, 0])
        return x, y, theta


class TagLocalizer:
    """Finds the robot pose from the newest fresh AprilTag detection.

    Applies three filters, in order:
    1. Age — drop detections older than MAX_TAG_AGE.
    2. Motion — drop detections from before the robot's last movement.
    3. Outlier — drop a reading that disagrees with the last good one
                 (same tag only; a tag switch is a legitimate re-anchor).
    If outlier filter rejects for too long, the baseline resets.
    """

    def __init__(self, node, estimator, tag_ids):
        self.node = node
        self.estimator = estimator
        self.tag_ids = tag_ids
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.last_pose_stamp = None
        self.last_good_pose = None
        self.last_good_tag_id = None
        self.last_accept_time = None

    def get_pose(self, motion_stamp_ns):
        """Return (x, y, theta, tag_id) or None."""
        for tag_id in self.tag_ids:
            try:
                tf = self.buffer.lookup_transform(
                    CAMERA_FRAME,
                    f"{TAG_FRAME_PREFIX}{tag_id}",
                    Time(),
                )
            except Exception:
                continue

            stamp_ns = Time.from_msg(tf.header.stamp).nanoseconds
            age = (
                self.node.get_clock().now()
                - Time.from_msg(tf.header.stamp)
            ).nanoseconds / 1e9
            if age < 0 or age > MAX_TAG_AGE:
                continue

            if motion_stamp_ns is not None and stamp_ns <= motion_stamp_ns:
                continue
            if (self.last_pose_stamp is not None
                    and stamp_ns <= self.last_pose_stamp):
                continue

            t = tf.transform.translation
            q = tf.transform.rotation
            pose = self.estimator.estimate(
                tag_id,
                [t.x, t.y, t.z],
                [q.x, q.y, q.z, q.w],
            )
            if not all(math.isfinite(v) for v in pose):
                continue

            if self._is_outlier(pose, tag_id):
                continue

            self._accept(pose, tag_id, stamp_ns)
            return (*pose, tag_id)

        return None

    def _is_outlier(self, pose, tag_id):
        # Timeout escape: a stuck baseline is probably stale.
        now = time.monotonic()
        if (self.last_accept_time is not None
                and now - self.last_accept_time > FILTER_RESET_TIMEOUT):
            self.node.get_logger().warn("Filter baseline stale, resetting.")
            self.last_good_pose = None
            self.last_good_tag_id = None

        # Only the SAME tag triggers the outlier check.
        if (self.last_good_pose is None
                or tag_id != self.last_good_tag_id):
            return False

        dx = pose[0] - self.last_good_pose[0]
        dy = pose[1] - self.last_good_pose[1]
        dtheta = wrap(pose[2] - self.last_good_pose[2])
        if (math.hypot(dx, dy) > MAX_POSE_JUMP
                or abs(dtheta) > MAX_HEADING_JUMP):
            self.node.get_logger().warn(
                f"Rejected tag {tag_id} outlier: "
                f"d={math.hypot(dx, dy):.2f} m, "
                f"dtheta={math.degrees(dtheta):.1f} deg"
            )
            return True
        return False

    def _accept(self, pose, tag_id, stamp_ns):
        self.last_good_pose = pose
        self.last_good_tag_id = tag_id
        self.last_pose_stamp = stamp_ns
        self.last_accept_time = time.monotonic()
