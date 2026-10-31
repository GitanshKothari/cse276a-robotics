#!/usr/bin/env python3
"""Measure how far the rover actually turns, and suggest new turn constants.

The open-loop plan times its turns: a 90 degree turn is LEFT_CMD held for
LEFT_90_S seconds. If the rover overshoots, those seconds are too long.
Rather than guess, this spins the rover for the currently configured time
and reads the real angle off the IMU the motor node already publishes,
then prints the constant that would have produced exactly 90 degrees.

Run the motor node first, then:

    python3 calibrate_turns.py --dir left
    python3 calibration/calibrate_turns.py --dir right --turns 4

Give the rover clear floor space; it spins on the spot. --turns spins
several 90 degree units in one go, which spreads the IMU's own noise over
a bigger angle and gives a tighter estimate.
"""

import argparse
import math
import time

import rclpy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from robot_control.config import COMMAND_DT, LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TurnCalibrator:
    def __init__(self, node):
        self.node = node
        self.yaw = None
        self.total = 0.0
        self.accumulating = False
        node.create_subscription(Imu, 'imu_data', self.on_imu, 10)
        self.pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)

    def on_imu(self, msg):
        q = msg.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        # Unwrap as we go so a turn through +/-180 still adds up.
        if self.accumulating and self.yaw is not None:
            self.total += wrap(yaw - self.yaw)
        self.yaw = yaw

    def pump(self, seconds):
        """Spin the ROS loop for a while without commanding the motors."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def wait_for_imu(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.yaw is None and time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.yaw is not None

    def publish(self, left, right):
        self.pub.publish(Float32MultiArray(data=[float(left), float(right)]))

    def spin_rover(self, left, right, seconds):
        """Hold a wheel pair for a duration, keeping the watchdog fed."""
        self.total = 0.0
        self.accumulating = True
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            self.publish(left, right)
            # The motor node safety-stops if commands pause over 0.15 s.
            self.pump(COMMAND_DT)
        for _ in range(3):
            self.publish(0.0, 0.0)
            self.pump(COMMAND_DT)
        # Let the rover coast to a stop before reading the final angle.
        self.pump(1.0)
        self.accumulating = False
        return self.total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', choices=['left', 'right'], required=True,
                        help='which direction to calibrate')
    parser.add_argument('--turns', type=float, default=1.0,
                        help='how many 90 degree units to spin (default 1)')
    parser.add_argument('--cmd', type=float, default=None,
                        help='wheel command to use (default: the configured one)')
    parser.add_argument('--seconds', type=float, default=None,
                        help='override the per-90-degree time being tested')
    args = parser.parse_args()

    turning_left = args.dir == 'left'
    cmd = args.cmd if args.cmd is not None else (LEFT_CMD if turning_left else RIGHT_CMD)
    per_90 = args.seconds if args.seconds is not None else (
        LEFT_90_S if turning_left else RIGHT_90_S)
    name = 'LEFT_90_S' if turning_left else 'RIGHT_90_S'

    duration = per_90 * args.turns
    expected = 90.0 * args.turns
    # Matches add_turn() in trajectory_planner: left spins the wheels apart
    # one way, right the other.
    left, right = (-cmd, cmd) if turning_left else (cmd, -cmd)

    rclpy.init()
    node = rclpy.create_node('turn_calibrator')
    cal = TurnCalibrator(node)

    try:
        print(f'Testing {name} = {per_90:.3f} s at command {cmd:.2f}')
        print(f'Spinning {args.dir} for {duration:.2f} s, expecting {expected:.1f} deg')

        if not cal.wait_for_imu():
            raise SystemExit('No IMU data on /imu_data - is motor_control running?')

        print('Starting in 3 s - keep clear...')
        cal.pump(3.0)

        measured = math.degrees(cal.spin_rover(left, right, duration))

        print()
        print(f'Commanded: {expected:.1f} deg over {duration:.2f} s')
        print(f'Measured:  {measured:+.1f} deg  (IMU)')

        magnitude = abs(measured)
        print(f'Rate:      {magnitude / duration:.1f} deg/s '
              f'(expected {expected / duration:.1f})')

        # A timed turn can only be rescaled if the rover actually turned.
        # Far short of the commanded angle means the motors stalled or the
        # wheels slipped, and the ratio below would then "correct" a
        # mechanical problem by inventing an absurdly long turn time.
        if magnitude < 0.5 * expected:
            print()
            print('REFUSING to suggest a constant: the rover did not come')
            print(f'close to the commanded {expected:.0f} deg. Scaling the time')
            print('here would only paper over a hardware problem. Check:')
            print('  - battery charge (a sagging pack stalls the motors)')
            print('  - wheels touching the floor, with traction')
            print('  - nothing blocking the rover as it spins')
            print('Re-run once the rover visibly completes the full spin.')
            return

        error = magnitude - expected
        print(f'Error:     {error:+.1f} deg '
              f'({100.0 * error / expected:+.1f}%) '
              f'{"overshoot" if error > 0 else "undershoot"}')
        print()

        # Time scales linearly with angle, so correct by the ratio measured.
        suggested = per_90 * expected / magnitude
        print(f'Suggested {name} = {suggested:.3f}   (was {per_90:.3f})')
        if magnitude < 0.8 * expected or magnitude > 1.25 * expected:
            print('NOTE: that is a big correction - re-run to confirm it repeats')
            print('      before trusting it.')
        print('Edit robot_control/robot_control/config.py, then rebuild:')
        print('  colcon build --packages-select robot_control robot_navigation')
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            cal.publish(0.0, 0.0)
            time.sleep(COMMAND_DT)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
