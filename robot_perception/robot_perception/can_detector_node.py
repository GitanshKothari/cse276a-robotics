"""Publish coke cans as landmark observations.

    camera/image_raw -> can_detector -> can_detections (Detection2DArray)

Each detection carries the pixel box and, once camera_info arrives, the
can's position in the camera optical frame. The localizer folds a can in
with the same ``base_from_camera`` transform it uses for an AprilTag.

A can is a weaker landmark than a tag, and this node is tuned for that. A
cylinder has no orientation, so one can can never fix the rover's heading.
Its bearing is good, because it comes from where the blob sits in the
image, and that does not decay with distance. Its range is poor, because
it comes from apparent size, so the error grows with the square of the
distance. Hence the short range limit.

This node does input and output only. It finds nothing itself: it holds a
``Detector`` and asks it.
"""

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

from .camera_model import PinholeCamera
from .detector import Detection, Detector
from .hsv_can_detector import HsvCanConfig, HsvCanDetector

CLASS_ID = 'coke_can'

# Changing any of these rebuilds the detector, so the colour gates can be
# tuned with `ros2 param set` while the debug image is on screen.
TUNABLE = frozenset({
    'downscale', 'label_height', 'can_diameter', 'target_aspect',
    'hue_low_max', 'hue_high_min', 'saturation_min', 'value_min',
    'blur_kernel', 'open_kernel', 'close_kernel',
    'min_height_px', 'min_area_px', 'edge_margin_px',
    'aspect_tolerance', 'fill_min', 'max_detections',
})


class CanDetectorNode(Node):
    def __init__(self):
        super().__init__('can_detector')
        self._declare_parameters()

        self._scale = max(1, self.get_parameter('downscale').value)
        # Declared as the protocol, not the class: a learned detector can
        # take its place here and nothing else in this file changes.
        self._detector: Detector = HsvCanDetector(self._detector_config())
        self._camera: PinholeCamera | None = None
        self._bridge = CvBridge()
        self._stats = _Stats()

        self._detections = self.create_publisher(
            Detection2DArray, self.get_parameter('detections_topic').value, 10)
        self._debug = self.create_publisher(
            Image, self.get_parameter('debug_topic').value, 1)

        self.add_post_set_parameters_callback(self._on_parameters_set)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.on_camera_info, 10)
        # The raw stream is 2.76 MB a frame at 1280x720. Measured against
        # this camera, a best-effort subscriber gets 1.5 Hz of it and a
        # reliable one 6.5 Hz, while the JPEG stream arrives at 9.3 Hz for
        # about 100 kB a frame. The detector thresholds colour, which
        # survives JPEG easily, so it reads the compressed topic by default
        # and leaves the raw bytes to the tag detector.
        topic = self.get_parameter('image_topic').value
        if self.get_parameter('use_compressed').value:
            self.create_subscription(
                CompressedImage, f'{topic}/compressed',
                self.on_image, qos_profile_sensor_data)
        else:
            self.create_subscription(
                Image, topic, self.on_image, qos_profile_sensor_data)

        self.get_logger().info(
            f"label {1000 * self.get_parameter('label_height').value:.0f} mm, "
            f"aspect {self._detector.config.target_aspect:.2f}, "
            f"range limit {self.get_parameter('max_range').value:.2f} m")

    # ------------------------------------------------------------------
    # Parameters

    def _declare_parameters(self):
        self.declare_parameter('image_topic', 'camera/image_raw')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('camera_info_topic', 'camera/camera_info')
        self.declare_parameter('detections_topic', 'can_detections')
        self.declare_parameter('debug_topic', 'can_detector/debug_image')
        self.declare_parameter('publish_debug', True)

        # 2 searches a 640 x 360 frame, which keeps up with the 15 Hz camera.
        self.declare_parameter('downscale', 2)

        # label_height is the RED BAND, not the can. The ends are bare
        # aluminium and stay out of the blob, so the band is shorter than
        # the 110 mm can. It scales every range by the same factor, so
        # calibrate it against a measured distance, not a ruler.
        self.declare_parameter('label_height', 0.090)
        self.declare_parameter('can_diameter', 0.070)

        # The shape gate, measured rather than derived.
        #
        # Geometry says this should be label_height / can_diameter, since
        # the label wraps the cylinder and its projected width is the
        # diameter. On real frames it is not. A cylinder's edges curve away
        # from the camera, so they are dim and desaturated and fall outside
        # the red threshold however loosely it is set: dropping saturation
        # from 120 to 60 widened a real can only from 35 to 42 px, where
        # geometry wanted 54. Height does not suffer that way, because the
        # top and bottom edges of the label face the camera squarely.
        #
        # So a blob reads narrower, and therefore taller, than the can
        # really is, and the gate has to match what the detector sees.
        self.declare_parameter('target_aspect', 1.75)

        self.declare_parameter('min_range', 0.25)
        self.declare_parameter('max_range', 2.5)

        self.declare_parameter('hue_low_max', 10)
        self.declare_parameter('hue_high_min', 170)
        self.declare_parameter('saturation_min', 120)
        self.declare_parameter('value_min', 60)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('open_kernel', 3)
        self.declare_parameter('close_kernel', 9)

        # Pixel gates are given at full resolution and scaled down here, so
        # they do not change when downscale changes.
        self.declare_parameter('min_height_px', 24)
        self.declare_parameter('min_area_px', 480)
        self.declare_parameter('edge_margin_px', 4)

        self.declare_parameter('aspect_tolerance', 0.45)
        self.declare_parameter('fill_min', 0.50)
        self.declare_parameter('max_detections', 8)
        self.declare_parameter('report_period', 5.0)

    def _detector_config(self) -> HsvCanConfig:
        scale = self._scale
        return HsvCanConfig(
            hue_low_max=self.get_parameter('hue_low_max').value,
            hue_high_min=self.get_parameter('hue_high_min').value,
            saturation_min=self.get_parameter('saturation_min').value,
            value_min=self.get_parameter('value_min').value,
            blur_kernel=self.get_parameter('blur_kernel').value,
            open_kernel=self.get_parameter('open_kernel').value,
            close_kernel=self.get_parameter('close_kernel').value,
            min_height=max(4, int(self.get_parameter('min_height_px').value / scale)),
            min_area=max(16, int(self.get_parameter('min_area_px').value / (scale * scale))),
            target_aspect=self.get_parameter('target_aspect').value,
            aspect_tolerance=self.get_parameter('aspect_tolerance').value,
            fill_min=self.get_parameter('fill_min').value,
            edge_margin=max(1, int(self.get_parameter('edge_margin_px').value / scale)),
            max_detections=self.get_parameter('max_detections').value,
        )

    def _on_parameters_set(self, parameters):
        """Rebuild the detector after a threshold moves.

        The node asks a ``Detector`` for boxes and never tunes one itself,
        so a change of threshold means a new detector rather than a mutated
        one. The config stays immutable, which keeps every frame consistent.
        """
        if not any(parameter.name in TUNABLE for parameter in parameters):
            return
        self._scale = max(1, self.get_parameter('downscale').value)
        self._detector = HsvCanDetector(self._detector_config())
        self.get_logger().info(
            f'retuned: aspect {self._detector.config.target_aspect:.2f}, '
            f'saturation >= {self._detector.config.saturation_min}, '
            f'value >= {self._detector.config.value_min}')

    # ------------------------------------------------------------------
    # Inputs

    def on_camera_info(self, msg):
        camera = PinholeCamera.from_projection_matrix(msg.p)
        if not camera.is_valid:
            return
        if self._camera is None:
            self.get_logger().info(
                f'camera fx {camera.fx:.1f} fy {camera.fy:.1f} '
                f'cx {camera.cx:.1f} cy {camera.cy:.1f}')
        self._camera = camera

    def on_image(self, msg):
        started = time.monotonic()
        frame = self._to_bgr(msg)
        if frame is None:
            return

        searched = self._shrink(frame)
        found = self._detector.detect(searched)

        array = Detection2DArray()
        array.header = msg.header
        drawn = []

        for detection in found:
            depth = self._depth_of(detection)
            if depth is None and self._camera is not None:
                continue        # outside the usable range
            array.detections.append(self._to_message(msg.header, detection, depth))
            drawn.append((detection, depth))

        self._detections.publish(array)
        self._stats.add(len(array.detections), time.monotonic() - started)
        self._report()

        if self._wants_debug():
            self._publish_debug(searched, drawn, msg.header)

    def _to_bgr(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                return cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                                    cv2.IMREAD_COLOR)
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:                      # noqa: BLE001
            self.get_logger().warn(f'cannot convert frame: {error}',
                                   throttle_duration_sec=5.0)
            return None

    def _shrink(self, frame):
        if self._scale == 1:
            return frame
        size = (frame.shape[1] // self._scale, frame.shape[0] // self._scale)
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Geometry

    def _depth_of(self, detection: Detection) -> float | None:
        """Depth of one detection, or None without a camera or out of range."""
        if self._camera is None:
            return None
        full = detection.scaled(self._scale)
        depth = self._camera.depth_from_height(
            full.height, self.get_parameter('label_height').value)
        low = self.get_parameter('min_range').value
        high = self.get_parameter('max_range').value
        return depth if low <= depth <= high else None

    def _to_message(self, header, detection: Detection,
                    depth: float | None) -> Detection2D:
        full = detection.scaled(self._scale)
        message = Detection2D()
        message.header = header
        message.bbox = BoundingBox2D()
        message.bbox.center.position.x = float(full.center[0])
        message.bbox.center.position.y = float(full.center[1])
        message.bbox.size_x = float(full.width)
        message.bbox.size_y = float(full.height)

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = CLASS_ID
        hypothesis.hypothesis.score = float(detection.confidence)
        hypothesis.pose.pose.orientation.w = 1.0
        if depth is not None and self._camera is not None:
            x, y, z = self._camera.back_project(full.center, depth)
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y
            hypothesis.pose.pose.position.z = z
        message.results.append(hypothesis)
        return message

    # ------------------------------------------------------------------
    # Outputs

    def _wants_debug(self) -> bool:
        return (self.get_parameter('publish_debug').value
                and self._debug.get_subscription_count() > 0)

    def _publish_debug(self, searched, drawn, header):
        canvas = searched.copy()
        for detection, depth in drawn:
            corner = (detection.x, detection.y)
            opposite = (detection.x + detection.width,
                        detection.y + detection.height)
            cv2.rectangle(canvas, corner, opposite, (0, 255, 0), 2)
            label = f'{depth:.2f}m' if depth is not None else 'no info'
            cv2.putText(canvas, f'{label} c{detection.confidence:.2f}',
                        (detection.x, max(12, detection.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        try:
            message = self._bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            message.header = header
            self._debug.publish(message)
        except Exception as error:                      # noqa: BLE001
            self.get_logger().warn(f'cannot publish debug frame: {error}',
                                   throttle_duration_sec=5.0)

    def _report(self):
        period = self.get_parameter('report_period').value
        line = self._stats.due(period)
        if line:
            self.get_logger().info(line)
        if self._camera is None:
            self.get_logger().warn('no camera_info yet, so no range is published',
                                   throttle_duration_sec=10.0)


class _Stats:
    """Counts frames and detections, for a periodic log line."""

    def __init__(self):
        self._since = time.monotonic()
        self._frames = 0
        self._detections = 0
        self._seconds = 0.0

    def add(self, detections: int, seconds: float):
        self._frames += 1
        self._detections += detections
        self._seconds += seconds

    def due(self, period: float) -> str | None:
        now = time.monotonic()
        elapsed = now - self._since
        if period <= 0.0 or elapsed < period or not self._frames:
            return None
        line = (f'{self._frames / elapsed:.1f} frames/s, '
                f'{self._detections / self._frames:.2f} cans/frame, '
                f'{1000.0 * self._seconds / self._frames:.1f} ms/frame')
        self._since = now
        self._frames = 0
        self._detections = 0
        self._seconds = 0.0
        return line


def main(args=None):
    rclpy.init(args=args)
    node = CanDetectorNode()
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
