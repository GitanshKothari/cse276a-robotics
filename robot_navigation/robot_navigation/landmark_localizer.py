"""Fuse AprilTag sightings with commanded motion into a pose estimate.

Each frame of detections is turned into one range/bearing observation per
mapped tag. Those are the strong measurements: the bearing of a tag is good
to a fraction of a degree and does not decay with distance, and the range
is read off the tag's apparent size. The orientation the solver recovers
for a tag is an order of magnitude worse and is only brought in when a
single tag is all there is, because then nothing else observes heading.
"""

import json
import math
import time

import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Imu
from std_msgs.msg import Float32MultiArray, String
from vision_msgs.msg import Detection2DArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .ekf import PoseEKF
from .kinematics import wheels_to_twist
from .landmark_map import LandmarkMap
from .tag_pose import (
    range_bearing, range_bearing_noise, robot_pose_from_tag, solve_tag_pose,
    tag_in_base, yaw_noise,
)
from .transforms import (
    camera_extrinsics, planar_pose, quaternion_from_yaw, wrap_angle,
    yaw_from_quaternion,
)


class LandmarkLocalizer(Node):
    def __init__(self):
        super().__init__('landmark_localizer')

        self.declare_parameter('map_path', '')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_frame')

        # Where the camera sits on the rover, in base_link metres/radians.
        self.declare_parameter('camera_x', 0.10)
        self.declare_parameter('camera_y', 0.0)
        self.declare_parameter('camera_z', 0.18)
        self.declare_parameter('camera_yaw_deg', 0.0)
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_roll_deg', 0.0)

        self.declare_parameter('initial_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('initial_std', [0.03, 0.03, 0.05])
        self.declare_parameter('predict_rate', 20.0)

        # Motion noise: a fraction of the commanded speed plus a floor.
        self.declare_parameter('alpha_v', 0.25)
        self.declare_parameter('beta_v', 0.02)
        self.declare_parameter('alpha_w', 0.25)
        self.declare_parameter('beta_w', 0.05)
        self.declare_parameter('command_timeout', 0.15)

        # Detection quality gates.
        self.declare_parameter('pixel_sigma', 1.0)
        self.declare_parameter('min_decision_margin', 30.0)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('max_reprojection_error', 3.0)
        self.declare_parameter('range_floor', 0.005)
        self.declare_parameter('bearing_floor_deg', 0.3)
        self.declare_parameter('mahalanobis_gate', 16.0)

        # Tag orientation as a heading source.
        self.declare_parameter('use_single_tag_yaw', True)
        self.declare_parameter('yaw_sigma_scale', 10.0)
        self.declare_parameter('ambiguity_threshold', 0.7)
        self.declare_parameter('yaw_floor_deg', 1.0)
        self.declare_parameter('yaw_gate', 9.0)

        self.declare_parameter('use_imu_heading', True)
        self.declare_parameter('imu_yaw_sigma_deg', 8.0)

        # Coke cans, the custom landmark option. A can gives range and
        # bearing only: a cylinder has no orientation to recover, so one
        # can can never fix the heading and two must be in view before the
        # pose is observable. Its range comes from apparent size, like a
        # tag's, but the label is smaller and a colour edge is blurrier
        # than a tag corner, so can_pixel_sigma is several times larger and
        # can_max_range is correspondingly shorter.
        # Both landmark types reduce to the same range and bearing update,
        # so either can be switched off on its own. Turning tags off gives
        # the can-only run the assignment's second option asks for, with no
        # need to move anything in the room.
        self.declare_parameter('use_tags', True)
        self.declare_parameter('use_cans', True)
        self.declare_parameter('can_label_height', 0.090)
        self.declare_parameter('can_max_range', 2.4)
        self.declare_parameter('can_pixel_sigma', 3.0)
        self.declare_parameter('can_gate', 9.0)
        # How much worse the runner-up match must be before the best one is
        # trusted. Cans are identical, so an ambiguous sighting is thrown
        # away rather than guessed at.
        self.declare_parameter('can_ambiguity_ratio', 3.0)
        # If this many frames in a row show cans but match none of them,
        # the filter is lost and its covariance is lying about how lost.
        # Widening it reopens the gate so the next sighting can be used.
        self.declare_parameter('can_lost_frames', 15)
        self.declare_parameter('can_lost_position_std', 0.05)
        self.declare_parameter('can_lost_heading_deg', 4.0)

        self.world_frame = self.get_parameter('world_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.base_from_camera = camera_extrinsics(
            self.get_parameter('camera_x').value,
            self.get_parameter('camera_y').value,
            self.get_parameter('camera_z').value,
            math.radians(self.get_parameter('camera_yaw_deg').value),
            math.radians(self.get_parameter('camera_pitch_deg').value),
            math.radians(self.get_parameter('camera_roll_deg').value),
        )

        map_path = self.get_parameter('map_path').value
        if not map_path:
            raise RuntimeError('landmark_localizer needs a map_path parameter')
        self.landmark_map = LandmarkMap.from_file(map_path)
        self.get_logger().info(
            f'Loaded {len(self.landmark_map)} landmarks: {self.landmark_map.ids()}')
        if self.landmark_map.cans:
            self.get_logger().info(
                f'Loaded {len(self.landmark_map.cans)} cans: {self.landmark_map.can_ids()}')

        initial_pose = list(self.get_parameter('initial_pose').value)
        self.filter = PoseEKF(
            initial_pose=initial_pose,
            initial_std=list(self.get_parameter('initial_std').value),
            alpha_v=self.get_parameter('alpha_v').value,
            beta_v=self.get_parameter('beta_v').value,
            alpha_w=self.get_parameter('alpha_w').value,
            beta_w=self.get_parameter('beta_w').value,
        )

        self.projection = None
        self.command = (0.0, 0.0)
        self.command_stamp = 0.0
        self.last_predict = time.monotonic()
        self.imu_offset = None
        self.last_fix_time = None
        self.tags_in_view = 0
        self.cans_in_view = 0
        self.can_misses = 0
        self.rejected = 0

        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(CameraInfo, 'camera/camera_info', self.on_camera_info, 10)
        self.create_subscription(AprilTagDetectionArray, 'detections', self.on_detections, 10)
        self.create_subscription(Detection2DArray, 'can_detections', self.on_can_detections, 10)
        self.create_subscription(Float32MultiArray, 'motor_commands', self.on_command, 10)
        self.create_subscription(Imu, 'imu_data', self.on_imu, sensor_qos)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'robot_pose', 10)
        self.status_pub = self.create_publisher(String, 'localization_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'landmark_markers', 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        self.publish_static_transform()
        self.publish_landmark_markers()

        period = 1.0 / max(self.get_parameter('predict_rate').value, 1.0)
        self.create_timer(period, self.on_timer)
        self.create_timer(2.0, self.publish_landmark_markers)

        self.get_logger().info('Landmark localizer ready')

    # ------------------------------------------------------------------
    # Inputs

    def on_camera_info(self, msg):
        # apriltag_node reads the rectified intrinsics out of P, so match it.
        if msg.p[0] > 0.0 and msg.p[5] > 0.0:
            self.projection = (msg.p[0], msg.p[5], msg.p[2], msg.p[6])

    def on_command(self, msg):
        if len(msg.data) >= 2:
            self.command = wheels_to_twist(float(msg.data[0]), float(msg.data[1]))
            self.command_stamp = time.monotonic()

    def on_imu(self, msg):
        if not self.get_parameter('use_imu_heading').value:
            return
        orientation = msg.orientation
        yaw = yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)

        if self.imu_offset is None:
            # The IMU's zero is arbitrary, so anchor it to the pose the
            # filter was started at and use it for changes from there on.
            self.imu_offset = wrap_angle(self.filter.state[2] - yaw)
            return

        sigma = math.radians(self.get_parameter('imu_yaw_sigma_deg').value)
        self.filter.update_heading(wrap_angle(yaw + self.imu_offset), sigma ** 2,
                                   gate=self.get_parameter('mahalanobis_gate').value)

    def on_detections(self, msg):
        if self.projection is None:
            return
        if not self.get_parameter('use_tags').value:
            self.tags_in_view = 0
            return

        fx, fy, cx, cy = self.projection
        focal = 0.5 * (fx + fy)
        pixel_sigma = self.get_parameter('pixel_sigma').value
        max_range = self.get_parameter('max_range').value
        min_margin = self.get_parameter('min_decision_margin').value
        max_error = self.get_parameter('max_reprojection_error').value

        observations = []
        rejected = 0

        for detection in msg.detections:
            landmark = self.landmark_map.get(detection.id)
            if landmark is None:
                continue
            if detection.decision_margin < min_margin:
                rejected += 1
                continue

            corners = [(corner.x, corner.y) for corner in detection.corners]
            solved = solve_tag_pose(corners, landmark.size, fx, fy, cx, cy)
            if solved is None:
                rejected += 1
                continue

            transform, error, ambiguity = solved
            distance = float(np.linalg.norm(transform[:3, 3]))
            if error > max_error or distance > max_range or distance < 1e-3:
                rejected += 1
                continue

            observations.append((landmark, transform, distance, ambiguity))

        self.tags_in_view = len(observations)
        self.rejected = rejected
        if not observations:
            return

        gate = self.get_parameter('mahalanobis_gate').value
        range_floor = self.get_parameter('range_floor').value
        bearing_floor = math.radians(self.get_parameter('bearing_floor_deg').value)
        accepted = 0

        for landmark, transform, distance, _ in observations:
            position = tag_in_base(transform, self.base_from_camera)
            measured_range, measured_bearing = range_bearing(position)
            noise = range_bearing_noise(distance, focal, landmark.size,
                                        pixel_sigma, range_floor, bearing_floor)
            result = self.filter.update_range_bearing(
                landmark.position[:2], (measured_range, measured_bearing), noise, gate=gate)
            if result is not None:
                accepted += 1

        # Range and bearing to a single landmark leave the heading free, so
        # fall back on the tag's own orientation in that case only - with
        # two or more tags the bearings fix the heading far better than a
        # recovered tag orientation ever would.
        if (accepted and len(observations) == 1
                and self.get_parameter('use_single_tag_yaw').value):
            landmark, transform, distance, ambiguity = observations[0]
            if ambiguity <= self.get_parameter('ambiguity_threshold').value:
                heading = planar_pose(
                    robot_pose_from_tag(transform, landmark.pose, self.base_from_camera))[2]
                variance = yaw_noise(
                    distance, focal, landmark.size, pixel_sigma,
                    self.get_parameter('yaw_sigma_scale').value,
                    ambiguity, self.get_parameter('ambiguity_threshold').value,
                    math.radians(self.get_parameter('yaw_floor_deg').value))
                self.filter.update_heading(
                    heading, variance, gate=self.get_parameter('yaw_gate').value)

        if accepted:
            self.last_fix_time = time.monotonic()

    def on_can_detections(self, msg):
        """Fuse coke can sightings.

        This differs from the tag path in one way that matters. A tag
        announces which landmark it is, so the filter never has to ask. A
        can announces nothing: every can in the room looks the same. So
        each sighting is scored against every mapped can, and the best
        match is used only when the runner-up is clearly worse. An
        ambiguous sighting is thrown away. Matching a can to the wrong
        neighbour drags the estimate towards a landmark that is not there,
        and that makes the next match wronger still.

        The detector already reports the can in the camera optical frame,
        so ``base_from_camera`` applies here exactly as it does for a tag.
        """
        if not self.get_parameter('use_cans').value or not self.landmark_map.cans:
            return
        if self.projection is None:
            return

        fx, fy, _, _ = self.projection
        focal = 0.5 * (fx + fy)
        label = self.get_parameter('can_label_height').value
        max_range = self.get_parameter('can_max_range').value
        pixel_sigma = self.get_parameter('can_pixel_sigma').value
        gate = self.get_parameter('can_gate').value
        ratio = self.get_parameter('can_ambiguity_ratio').value
        range_floor = self.get_parameter('range_floor').value
        bearing_floor = math.radians(self.get_parameter('bearing_floor_deg').value)

        seen = 0
        accepted = 0

        for detection in msg.detections:
            if not detection.results:
                continue
            point = detection.results[0].pose.pose.position
            if point.z <= 0.0:
                continue        # the detector had no camera_info yet

            in_base = self.base_from_camera @ np.array(
                [point.x, point.y, point.z, 1.0])
            measured = range_bearing(in_base[:3])
            if not (0.05 < measured[0] <= max_range):
                continue
            seen += 1

            noise = range_bearing_noise(measured[0], focal, label, pixel_sigma,
                                        range_floor, bearing_floor)
            scored = []
            for can_id, can_xy in self.landmark_map.cans.items():
                distance = self.filter.range_bearing_mahalanobis(
                    can_xy, measured, noise)
                if distance is not None:
                    scored.append((distance, can_id, can_xy))
            if not scored:
                continue

            scored.sort(key=lambda item: item[0])
            best, best_id, best_xy = scored[0]
            if best > gate:
                continue
            if len(scored) > 1 and scored[1][0] < ratio * best:
                self.get_logger().debug(
                    f'can sighting at {measured[0]:.2f} m is ambiguous between '
                    f'{best_id} and {scored[1][1]}, dropped')
                continue

            if self.filter.update_range_bearing(
                    best_xy, measured, noise, gate=gate) is not None:
                accepted += 1

        self.cans_in_view = seen
        if accepted:
            self.can_misses = 0
            self.last_fix_time = time.monotonic()
        elif seen:
            # Cans are in view but none of them matched. A few frames of
            # that is noise; a run of it means the filter is lost.
            self.can_misses += 1
            if self.can_misses >= self.get_parameter('can_lost_frames').value:
                self.filter.inflate(
                    self.get_parameter('can_lost_position_std').value,
                    math.radians(self.get_parameter('can_lost_heading_deg').value))
                self.can_misses = 0
                self.get_logger().warn(
                    'cans in view but none matched, widening the covariance',
                    throttle_duration_sec=2.0)

    # ------------------------------------------------------------------
    # Outputs

    def on_timer(self):
        now = time.monotonic()
        dt = now - self.last_predict
        self.last_predict = now

        # The motor node stops the rover when commands stop arriving, so the
        # filter has to assume the same thing rather than coast forever.
        if now - self.command_stamp > self.get_parameter('command_timeout').value:
            self.command = (0.0, 0.0)

        self.filter.predict(self.command[0], self.command[1], dt)
        self.publish_pose()
        self.publish_status(now)

    def publish_pose(self):
        x, y, theta = self.filter.pose
        stamp = self.get_clock().now().to_msg()

        message = PoseWithCovarianceStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.world_frame
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(theta)
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw

        # The 6x6 ROS covariance only has x, y and yaw filled in.
        covariance = [0.0] * 36
        indices = (0, 1, 5)
        for row in range(3):
            for column in range(3):
                covariance[indices[row] * 6 + indices[column]] = float(
                    self.filter.covariance[row, column])
        message.pose.covariance = covariance
        self.pose_pub.publish(message)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def publish_status(self, now):
        blind = None if self.last_fix_time is None else now - self.last_fix_time
        x, y, theta = self.filter.pose
        payload = {
            'x': round(x, 4),
            'y': round(y, 4),
            'theta': round(theta, 4),
            'position_std': round(self.filter.position_std, 4),
            'heading_std': round(self.filter.heading_std, 4),
            'tags_in_view': self.tags_in_view,
            'cans_in_view': self.cans_in_view,
            'rejected': self.rejected,
            'seconds_since_fix': None if blind is None else round(blind, 2),
        }
        self.status_pub.publish(String(data=json.dumps(payload)))

    def publish_static_transform(self):
        rotation = self.base_from_camera[:3, :3]
        trace = float(np.trace(rotation))
        # Shepperd's method, picking the largest denominator for stability.
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = self.camera_frame
        transform.transform.translation.x = float(self.base_from_camera[0, 3])
        transform.transform.translation.y = float(self.base_from_camera[1, 3])
        transform.transform.translation.z = float(self.base_from_camera[2, 3])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.static_broadcaster.sendTransform(transform)

    def publish_landmark_markers(self):
        markers = MarkerArray()
        for index, tag_id in enumerate(self.landmark_map.ids()):
            landmark = self.landmark_map.get(tag_id)
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'landmarks'
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(landmark.position[0])
            marker.pose.position.y = float(landmark.position[1])
            marker.pose.position.z = float(landmark.position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = landmark.size
            marker.scale.y = landmark.size
            marker.scale.z = 0.01
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkLocalizer()
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
