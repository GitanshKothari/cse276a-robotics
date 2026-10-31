"""Software-in-the-loop stand-in for the rover and its camera.

This exists so the closed loop can be exercised without the hardware. It
subscribes to the same ``motor_commands`` the real motor node listens on,
integrates a unicycle with deliberately imperfect wheels, and synthesises
the ``detections`` and ``camera_info`` the real detector would publish by
projecting the mapped tags through the rectified camera model. The
localizer and the controller run unmodified against it.

What it is good for: catching frame and sign errors, checking that the
controller converges and that the filter stays consistent, and measuring
how the estimate compares with truth - which the real run cannot tell you
without hand-measuring the floor. What it is not: a substitute for the
hardware run. Motion blur, rolling shutter, exposure, battery sag and the
real floor are all absent, so the numbers it produces are optimistic.

The true pose goes out on ``true_pose`` so the logger can score against it.
"""

import math
import time

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Imu
from std_msgs.msg import Float32MultiArray

from .coverage_check import (
    DEFAULT_HEIGHT, DEFAULT_WIDTH, rectified_intrinsics, tag_is_visible,
)
from .kinematics import wheels_to_twist
from .landmark_map import LandmarkMap
from .tag_pose import object_points
from .transforms import (
    camera_extrinsics, invert_transform, make_transform, quaternion_from_yaw,
    rot_z, wrap_angle,
)


class SimRover(Node):
    def __init__(self):
        super().__init__('sim_rover')

        self.declare_parameter('map_path', '')
        self.declare_parameter('camera_config', '')
        self.declare_parameter('width', DEFAULT_WIDTH)
        self.declare_parameter('height', DEFAULT_HEIGHT)

        # Must match the localizer's idea of where the camera is, or the
        # test is measuring a disagreement about the mount rather than the
        # thing under test.
        self.declare_parameter('camera_x', 0.10)
        self.declare_parameter('camera_y', 0.0)
        self.declare_parameter('camera_z', 0.18)
        self.declare_parameter('camera_yaw_deg', 0.0)
        self.declare_parameter('camera_pitch_deg', 0.0)

        self.declare_parameter('start_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('physics_rate', 100.0)
        self.declare_parameter('detection_rate', 15.0)
        self.declare_parameter('imu_rate', 50.0)

        # The errors that make the open-loop version of this drift. The
        # scale factors are what a hand calibration gets wrong; the noise
        # is wheel slip frame to frame.
        self.declare_parameter('linear_scale', 0.92)
        self.declare_parameter('angular_scale', 1.08)
        self.declare_parameter('linear_noise', 0.04)
        self.declare_parameter('angular_noise', 0.06)
        self.declare_parameter('command_timeout', 0.15)

        self.declare_parameter('pixel_sigma', 0.7)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('max_oblique_deg', 65.0)
        self.declare_parameter('decision_margin', 60.0)
        self.declare_parameter('dropout_probability', 0.0)

        self.declare_parameter('imu_yaw_sigma_deg', 3.0)
        self.declare_parameter('imu_yaw_drift_deg_per_s', 0.2)
        self.declare_parameter('seed', 1)

        map_path = self.get_parameter('map_path').value
        if not map_path:
            raise RuntimeError('sim_rover needs a map_path parameter')
        self.landmark_map = LandmarkMap.from_file(map_path)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.intrinsics = rectified_intrinsics(
            self.get_parameter('camera_config').value, self.width, self.height)

        self.base_from_camera = camera_extrinsics(
            self.get_parameter('camera_x').value,
            self.get_parameter('camera_y').value,
            self.get_parameter('camera_z').value,
            math.radians(self.get_parameter('camera_yaw_deg').value),
            math.radians(self.get_parameter('camera_pitch_deg').value),
            0.0,
        )

        self.pose = np.array(self.get_parameter('start_pose').value, dtype=float)
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))
        self.command = (0.0, 0.0)
        self.command_stamp = 0.0
        self.last_step = time.monotonic()
        self.imu_drift = 0.0

        self.create_subscription(
            Float32MultiArray, 'motor_commands', self.on_command, 10)
        self.detection_pub = self.create_publisher(
            AprilTagDetectionArray, 'detections', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu_data', 10)
        self.truth_pub = self.create_publisher(PoseStamped, 'true_pose', 10)

        self.create_timer(1.0 / self.get_parameter('physics_rate').value, self.step)
        self.create_timer(1.0 / self.get_parameter('detection_rate').value, self.emit_frame)
        self.create_timer(1.0 / self.get_parameter('imu_rate').value, self.emit_imu)
        self.create_timer(1.0, self.emit_camera_info)
        self.emit_camera_info()

        self.get_logger().info(
            f'Simulating {len(self.landmark_map)} tags, '
            f'start pose {tuple(round(v, 3) for v in self.pose)}')

    # ------------------------------------------------------------------

    def on_command(self, msg):
        if len(msg.data) >= 2:
            self.command = wheels_to_twist(float(msg.data[0]), float(msg.data[1]))
            self.command_stamp = time.monotonic()

    def step(self):
        now = time.monotonic()
        dt = now - self.last_step
        self.last_step = now
        if dt <= 0.0:
            return

        # The real motor node stops on a command gap, so this has to too.
        if now - self.command_stamp > self.get_parameter('command_timeout').value:
            self.command = (0.0, 0.0)

        v, omega = self.command
        if v or omega:
            v *= self.get_parameter('linear_scale').value * (
                1.0 + self.rng.normal(0.0, self.get_parameter('linear_noise').value))
            omega *= self.get_parameter('angular_scale').value * (
                1.0 + self.rng.normal(0.0, self.get_parameter('angular_noise').value))

        mid = self.pose[2] + 0.5 * omega * dt
        self.pose[0] += v * dt * math.cos(mid)
        self.pose[1] += v * dt * math.sin(mid)
        self.pose[2] = wrap_angle(self.pose[2] + omega * dt)
        self.imu_drift += math.radians(
            self.get_parameter('imu_yaw_drift_deg_per_s').value) * dt

        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'world'
        message.pose.position.x = float(self.pose[0])
        message.pose.position.y = float(self.pose[1])
        qx, qy, qz, qw = quaternion_from_yaw(float(self.pose[2]))
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        self.truth_pub.publish(message)

    # ------------------------------------------------------------------

    def emit_camera_info(self):
        message = CameraInfo()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'camera_frame'
        message.width = self.width
        message.height = self.height
        message.distortion_model = 'plumb_bob'
        message.d = [0.0] * 5
        message.k = [float(v) for v in self.intrinsics.reshape(-1)]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        projection = np.zeros((3, 4))
        projection[:3, :3] = self.intrinsics
        message.p = [float(v) for v in projection.reshape(-1)]
        self.info_pub.publish(message)

    def emit_frame(self):
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        pixel_sigma = self.get_parameter('pixel_sigma').value
        dropout = self.get_parameter('dropout_probability').value

        world_from_base = make_transform(
            rot_z(self.pose[2]), (self.pose[0], self.pose[1], 0.0))
        inverse = invert_transform(self.base_from_camera) @ invert_transform(world_from_base)

        array = AprilTagDetectionArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.header.frame_id = 'camera_frame'

        pose = (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))
        for tag_id in self.landmark_map.ids():
            landmark = self.landmark_map.get(tag_id)
            if not tag_is_visible(pose, landmark.pose, landmark.size,
                                  self.base_from_camera, self.intrinsics,
                                  self.width, self.height,
                                  self.get_parameter('max_range').value,
                                  self.get_parameter('max_oblique_deg').value):
                continue
            if dropout and self.rng.random() < dropout:
                continue

            camera_from_tag = inverse @ landmark.pose
            detection = AprilTagDetection()
            detection.family = '36h11'
            detection.id = int(tag_id)
            detection.hamming = 0
            detection.decision_margin = float(self.get_parameter('decision_margin').value)

            centre = camera_from_tag[:3, 3]
            detection.centre.x = float(fx * centre[0] / centre[2] + cx)
            detection.centre.y = float(fy * centre[1] / centre[2] + cy)

            for index, point in enumerate(object_points(landmark.size)):
                corner = camera_from_tag @ np.append(point, 1.0)
                u = fx * corner[0] / corner[2] + cx
                v = fy * corner[1] / corner[2] + cy
                if pixel_sigma:
                    u += self.rng.normal(0.0, pixel_sigma)
                    v += self.rng.normal(0.0, pixel_sigma)
                detection.corners[index].x = float(u)
                detection.corners[index].y = float(v)

            array.detections.append(detection)

        self.detection_pub.publish(array)

    def emit_imu(self):
        sigma = math.radians(self.get_parameter('imu_yaw_sigma_deg').value)
        yaw = wrap_angle(float(self.pose[2]) + self.imu_drift
                         + self.rng.normal(0.0, sigma))
        message = Imu()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'imu_link'
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        message.orientation.x = qx
        message.orientation.y = qy
        message.orientation.z = qz
        message.orientation.w = qw
        self.imu_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SimRover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
