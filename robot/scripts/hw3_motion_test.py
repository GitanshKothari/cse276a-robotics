#!/usr/bin/env python3
"""Calibration test for the HW3 turn pulses and forward steps.

Uses the same wheel commands and timing as robot/robot/hw3_run.py.

  hw3_motion_test.py left                     one 90 deg left turn
  hw3_motion_test.py left --pulses 4          90 deg in 4 pulses (as in the run)
  hw3_motion_test.py right --pulses 4
  hw3_motion_test.py left --pulses 4 --seconds-90 0.78   try a new calibration
  hw3_motion_test.py forward --pulses 10      ten 0.10 m steps (1.00 m)
  hw3_motion_test.py forward --pulses 10 --speed 0.21    try a new step speed
"""
import argparse
import time

import rclpy
from std_msgs.msg import Float32MultiArray

from robot.config import COMMAND_DT, FORWARD_CMD
from robot.config import LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S
from robot.hw3_run import STEP_M, STEP_SPEED, TURN_CAL_DEG, TURN_LAG_S


def pulse_seconds(angle_deg, seconds_90):
    """Same turn pulse model as hw3_run.turn_to."""
    rate = TURN_CAL_DEG / (TURN_CAL_DEG / 90.0 * seconds_90 - TURN_LAG_S)
    return TURN_LAG_S + angle_deg / rate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("motion", choices=["left", "right", "forward"])
    p.add_argument("--angle", type=float, default=90.0,
                   help="total turn angle in degrees (left/right)")
    p.add_argument("--pulses", type=int, default=1,
                   help="number of equal pulses, with a stop after each")
    p.add_argument("--seconds-90", type=float, default=None,
                   help="override LEFT_90_S / RIGHT_90_S for this test")
    p.add_argument("--step-m", type=float, default=STEP_M,
                   help="length of one forward pulse in metres")
    p.add_argument("--speed", type=float, default=None,
                   help="override STEP_SPEED for this test (m/s)")
    p.add_argument("--pause", type=float, default=1.0,
                   help="stop time after each pulse in seconds")
    args = p.parse_args()
    if args.pulses < 1:
        p.error("pulses must be 1 or more")

    if args.motion == "left":
        s90 = args.seconds_90 if args.seconds_90 else LEFT_90_S
        wheels = (-LEFT_CMD, LEFT_CMD)
        duration = pulse_seconds(args.angle / args.pulses, s90)
        what = f"left {args.angle / args.pulses:.1f} deg, LEFT_90_S={s90:.3f}"
    elif args.motion == "right":
        s90 = args.seconds_90 if args.seconds_90 else RIGHT_90_S
        wheels = (RIGHT_CMD, -RIGHT_CMD)
        duration = pulse_seconds(args.angle / args.pulses, s90)
        what = f"right {args.angle / args.pulses:.1f} deg, RIGHT_90_S={s90:.3f}"
    else:
        speed = args.speed if args.speed else STEP_SPEED
        wheels = (FORWARD_CMD, FORWARD_CMD)
        duration = args.step_m / speed
        what = f"forward {args.step_m:.2f} m, STEP_SPEED={speed:.3f}"

    rclpy.init()
    node = rclpy.create_node("hw3_motion_test")
    pub = node.create_publisher(Float32MultiArray, "/motor_commands", 10)

    def cmd(left, right):
        pub.publish(Float32MultiArray(data=[float(left), float(right)]))

    def stop():
        for _ in range(6):
            cmd(0.0, 0.0)
            time.sleep(COMMAND_DT)

    try:
        # Wait until discovery is stable, then count the motor nodes.
        t0 = time.monotonic()
        while pub.get_subscription_count() == 0:
            if time.monotonic() - t0 > 5:
                raise SystemExit("ERROR: no motor node. Start hw3.launch.py first.")
            time.sleep(0.1)
        time.sleep(1.0)
        count = pub.get_subscription_count()
        if count != 1:
            raise SystemExit(
                f"ERROR: {count} motor nodes listen on /motor_commands. "
                "Stop all launches and start only one.")

        print(f"{args.pulses} pulse(s): {what}, {duration:.3f} s each", flush=True)
        stop()
        for i in range(args.pulses):
            t0 = time.monotonic()
            while True:
                left = duration - (time.monotonic() - t0)
                if left <= 0:
                    break
                cmd(*wheels)
                time.sleep(min(COMMAND_DT, left))
            stop()
            print(f"  pulse {i + 1}: {time.monotonic() - t0 - 6 * COMMAND_DT:.3f} s",
                  flush=True)
            time.sleep(args.pause)
    finally:
        if rclpy.ok():
            stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
