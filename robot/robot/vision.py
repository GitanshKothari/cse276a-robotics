
import math

import numpy as np
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .config import CAMERA_FRAME, TAG_FRAME_PREFIX, MAX_TAG_AGE


def wrap(angle):
    # wrap the angle to be beteen -180 to 180
    return math.atan2(math.sin(angle), math.cos(angle))


def make_transform(position, quaternion):
    # making the transformation matrix given the quartenion
    x, y, z, w = quaternion

    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ]
    T[:3, 3] = position

    return T


class PoseEstimator:
    def __init__(self, landmarks, camera_mount):
        self.landmarks = landmarks
        self.camera_transform = make_transform(
            camera_mount["position"],
            camera_mount["rotation"]
        )

    def estimate(self, tag_id, position, quaternion):
        # calculating the world robot transformation based on the other transformations. 
        tag = self.landmarks[tag_id]

        world_tag = make_transform(
            tag["position"], tag["rotation"]
        )
        camera_tag = make_transform(position, quaternion)

        world_robot = (
            world_tag
            @ np.linalg.inv(camera_tag)
            @ np.linalg.inv(self.camera_transform)
        )

        x = world_robot[0, 3]
        y = world_robot[1, 3]
        theta = math.atan2(world_robot[1, 0], world_robot[0, 0])

        return x, y, theta


class TagLocalizer:
    def __init__(self, node, estimator, tag_ids):
        self.node = node
        self.estimator = estimator
        self.tag_ids = tag_ids

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

        self.last_pose_stamp = None

    def get_pose(self, motion_stamp_ns):
        for tag_id in self.tag_ids: #because of this loop and using the sorted array in navigator, we make sure we detect the tags in the priority order. 
            try:
                tf = self.buffer.lookup_transform(
                    CAMERA_FRAME,
                    f"{TAG_FRAME_PREFIX}{tag_id}",
                    Time()
                )
            except Exception:
                continue

            stamp = Time.from_msg(tf.header.stamp).nanoseconds
            now = self.node.get_clock().now().nanoseconds
            age = (now - stamp) / 1e9

            if age < 0 or age > MAX_TAG_AGE: # dont want stale data
                continue

            if motion_stamp_ns is not None and stamp <= motion_stamp_ns:
                continue

            if self.last_pose_stamp is not None and stamp <= self.last_pose_stamp:
                continue

            t = tf.transform.translation
            q = tf.transform.rotation

            pose = self.estimator.estimate(
                tag_id,
                [t.x, t.y, t.z],
                [q.x, q.y, q.z, q.w]
            )

            if not all(math.isfinite(v) for v in pose): # checks.
                continue

            self.last_pose_stamp = stamp
            return (*pose, tag_id)

        return None
