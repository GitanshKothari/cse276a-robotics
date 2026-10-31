#!/usr/bin/env python3
"""Drive each wheel on its own and measure the yaw it produces.

A spin needs both wheels pushing opposite ways, so one weak motor shows up
as a slow spin without saying which side is at fault. Driving a single
wheel pivots the rover around the other one, which gives each motor its
own number: a healthy wheel produces an obvious yaw rate, a dead one
produces almost none.

The rover arcs across the floor on each test rather than staying put, so
give it a couple of metres of clear space.

    python3 calibration/wheel_test.py
    python3 calibration/wheel_test.py --cmd 0.45 --seconds 1.0
"""

import argparse
import math
import sys
import time

import rclpy

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from calibrate_turns import TurnCalibrator


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cmd', type=float, default=0.40,
                        help='wheel command to test with (default 0.40)')
    parser.add_argument('--seconds', type=float, default=1.2,
                        help='how long to drive each wheel (default 1.2)')
    args = parser.parse_args()

    cmd = abs(args.cmd)
    tests = [
        ('left  wheel forward', cmd, 0.0),
        ('left  wheel reverse', -cmd, 0.0),
        ('right wheel forward', 0.0, cmd),
        ('right wheel reverse', 0.0, -cmd),
    ]

    rclpy.init()
    node = rclpy.create_node('wheel_test')
    cal = TurnCalibrator(node)
    results = []

    try:
        if not cal.wait_for_imu():
            raise SystemExit('No IMU data on /imu_data - is motor_control running?')

        print(f'Testing each wheel at command {cmd:.2f} for {args.seconds:.1f} s.')
        print('The rover will arc across the floor four times - keep it clear.')
        print('Starting in 5 s...')
        cal.pump(5.0)

        for label, left, right in tests:
            print(f'  {label} ... ', end='', flush=True)
            measured = math.degrees(cal.spin_rover(left, right, args.seconds))
            rate = abs(measured) / args.seconds
            results.append((label, measured, rate))
            print(f'{measured:+8.1f} deg  ({rate:6.1f} deg/s)')
            # Pause between tests so each starts from rest.
            cal.pump(1.5)

        print()
        print('Summary')
        for label, measured, rate in results:
            verdict = 'DEAD/STALLED' if rate < 10.0 else ('weak' if rate < 40.0 else 'ok')
            print(f'  {label}: {rate:6.1f} deg/s   {verdict}')

        left_rates = [r for lbl, _, r in results if lbl.startswith('left')]
        right_rates = [r for lbl, _, r in results if lbl.startswith('right')]
        left_ok = min(left_rates) >= 10.0
        right_ok = min(right_rates) >= 10.0

        print()
        if left_ok and right_ok:
            imbalance = abs(sum(left_rates) - sum(right_rates)) / max(
                sum(left_rates), sum(right_rates))
            print(f'Both wheels drive. Side imbalance: {100.0 * imbalance:.0f}%')
            if imbalance > 0.25:
                print('That is a lot - the weaker side will bend every turn.')
            else:
                print('If the rover still will not spin, suspect traction or '
                      'battery sag under the load of both motors together.')
        elif left_ok:
            print('The RIGHT motor is not driving. Check its connector at the')
            print('driver board, and that the wheel turns freely by hand.')
        elif right_ok:
            print('The LEFT motor is not driving. Check its connector at the')
            print('driver board, and that the wheel turns freely by hand.')
        else:
            print('NEITHER wheel is driving. That points at the battery or the')
            print('driver board rather than one motor - check the pack voltage')
            print('under load, and that the rover is powered on, not just the Pi.')
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            cal.publish(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
