"""Say why the detector is or is not finding cans, from one live frame.

Grabs a frame off the camera, runs the real detector over it, and prints
every red blob with the gate that rejected it. Use it when `cans/frame`
reads zero and you need to know whether the fault is the light, the
placement, or a threshold.

    ros2 run robot_perception can_check
"""

import sys
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from .hsv_can_detector import HsvCanConfig, HsvCanDetector

FOCAL_Y = 636.4
LABEL = 0.090


FIELDS = ('hue_low_max', 'hue_high_min', 'saturation_min', 'value_min',
          'blur_kernel', 'open_kernel', 'close_kernel',
          'target_aspect', 'aspect_tolerance', 'fill_min')


def live_config(node, timeout=4.0):
    """The running detector's thresholds, so this tool agrees with it.

    Without this, `ros2 param set` would retune the detector while this
    tool kept reporting the built-in defaults, and the two would disagree
    about why a blob was rejected.
    """
    client = node.create_client(GetParameters, '/can_detector/get_parameters')
    if not client.wait_for_service(timeout_sec=timeout):
        return HsvCanConfig(), False
    future = client.call_async(GetParameters.Request(names=list(FIELDS)))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if future.result() is None:
        return HsvCanConfig(), False

    values = {}
    for name, value in zip(FIELDS, future.result().values):
        if value.type == ParameterType.PARAMETER_INTEGER:
            values[name] = value.integer_value
        elif value.type == ParameterType.PARAMETER_DOUBLE:
            values[name] = value.double_value
    # min_height and friends are scaled by downscale inside the node, so
    # they are left at the dataclass defaults here.
    return HsvCanConfig(**values), True


def grab(node, timeout=15.0):
    frames = []
    node.create_subscription(CompressedImage, 'camera/image_raw/compressed',
                             frames.append, qos_profile_sensor_data)
    end = time.monotonic() + timeout
    while time.monotonic() < end and not frames:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not frames:
        return None
    return cv2.imdecode(np.frombuffer(frames[-1].data, np.uint8), cv2.IMREAD_COLOR)


def main(args=None):
    rclpy.init()
    node = rclpy.create_node('can_check')
    config, live = live_config(node)
    frame = grab(node)
    node.destroy_node()
    rclpy.shutdown()

    if frame is None:
        print('No frame. Is the camera running?')
        return 1
    print('settings   ' + ('read from the running can_detector'
                           if live else 'built-in defaults (detector not running)'))
    print(f'           aspect {config.target_aspect:.2f} +/- '
          f'{100*config.aspect_tolerance:.0f}%, fill >= {config.fill_min:.2f}, '
          f'sat >= {config.saturation_min}, value >= {config.value_min}')

    brightness = frame.mean()
    black = 100.0 * (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 25).mean()
    print(f'frame      {frame.shape[1]}x{frame.shape[0]}')
    print(f'brightness {brightness:.1f}/255   '
          f'{"TOO DARK - turn the lights on" if brightness < 55 else "ok"}')
    print(f'near black {black:.0f}% of the frame')

    detector = HsvCanDetector(config)
    mask = detector.red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'red mask   {100.0 * (mask > 0).mean():.3f}% of the frame')
    print(f'\n{len(contours)} red blobs, before the shape gates:')
    print(f'{"x":>5}{"y":>5}{"w":>5}{"h":>5}{"aspect":>8}{"fill":>6}   verdict')

    rows = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 1 or h < 1:
            continue
        area = cv2.contourArea(contour)
        aspect, fill = h / w, area / (w * h)
        why = []
        if h < config.min_height:
            why.append(f'too short ({h} < {config.min_height} px)')
        if area < config.min_area:
            why.append('too small')
        if y <= config.edge_margin or y + h >= frame.shape[0] - config.edge_margin:
            why.append('cut off by the frame edge')
        if abs(aspect - config.target_aspect) / config.target_aspect > config.aspect_tolerance:
            why.append(f'wrong shape (aspect {aspect:.2f}, want {config.target_aspect:.2f})')
        if fill < config.fill_min:
            why.append(f'too patchy (fill {fill:.2f})')
        verdict = ', '.join(why) if why else f'PASSES -> {FOCAL_Y * LABEL / h:.2f} m away'
        rows.append((area, x, y, w, h, aspect, fill, verdict))

    for area, x, y, w, h, aspect, fill, verdict in sorted(rows, reverse=True)[:10]:
        print(f'{x:>5}{y:>5}{w:>5}{h:>5}{aspect:>8.2f}{fill:>6.2f}   {verdict}')

    found = detector.detect(frame)
    print(f'\nDETECTOR FOUND {len(found)} CAN(S)')
    for d in found:
        print(f'  {d.width}x{d.height} px, confidence {d.confidence:.2f}, '
              f'{FOCAL_Y * LABEL / d.height:.2f} m away')
    if not found:
        print('  Nothing passed. Read the verdict column above.')
        if brightness < 55:
            print('  The frame is too dark. No threshold can fix that.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
