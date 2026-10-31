#!/usr/bin/env python3
"""Print the rover's IMU heading live, as degrees.

Useful for checking the IMU independently of the motors: turn the rover by
hand through a known angle and watch whether the numbers agree. 'total' is
the accumulated turn since start, so it keeps counting past 180 degrees
instead of wrapping.

    python3 calibration/watch_yaw.py
"""

import math
import time

import rclpy
from sensor_msgs.msg import Imu


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main():
    rclpy.init()
    node = rclpy.create_node('watch_yaw')
    state = {'yaw': None, 'total': 0.0, 'start': None}

    def on_imu(msg):
        q = msg.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        if state['yaw'] is not None:
            state['total'] += wrap(yaw - state['yaw'])
        else:
            state['start'] = yaw
        state['yaw'] = yaw

    node.create_subscription(Imu, 'imu_data', on_imu, 10)

    print('Waiting for /imu_data (Ctrl-C to stop)...')
    try:
        last = 0.0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if state['yaw'] is not None and now - last > 0.1:
                last = now
                print(f'\rheading {math.degrees(state["yaw"]):+8.2f} deg   '
                      f'total turned {math.degrees(state["total"]):+9.2f} deg',
                      end='', flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
