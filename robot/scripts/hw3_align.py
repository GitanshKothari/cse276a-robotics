#!/usr/bin/env python3
"""Placement check before a HW3 run. The robot does not move.

Put the camera on the start mark, then run this tool. It reads the tags that
the camera sees, compares them with the ground-truth CSV, and tells you how
to turn the robot so that it is parallel to the world x axis.

This tool is an operator aid only. The SLAM run does not use the ground truth.

  hw3_align.py truth.csv --start 0.9 0.9 0
"""
import argparse
import csv
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from robot.config import CAMERA_FRAME, TAG_FRAME_PREFIX
from robot.hw3_run import CAMERA_YAW_BIAS

FRONTAL_DEG = 15.0     # only tags this near the image centre give the heading
OK_DEG = 1.0
OK_RANGE_M = 0.03


class Align(Node):
    def __init__(self):
        super().__init__("hw3_align")
        self.views = {}
        self.create_subscription(TFMessage, "/tf", self.on_tf, 10)

    def on_tf(self, msg):
        for tf in msg.transforms:
            name = tf.child_frame_id
            if tf.header.frame_id != CAMERA_FRAME or not name.startswith(TAG_FRAME_PREFIX):
                continue
            p = tf.transform.translation
            fwd, left = float(p.z), float(-p.x)
            cb, sb = math.cos(CAMERA_YAW_BIAS), math.sin(CAMERA_YAW_BIAS)
            fwd, left = fwd * cb - left * sb, left * cb + fwd * sb
            self.views.setdefault(name[len(TAG_FRAME_PREFIX):], []).append((fwd, left))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("truth", help="ground-truth CSV (id,x,y)")
    p.add_argument("--start", type=float, nargs=3, default=[0.9, 0.9, 0.0],
                   metavar=("X_M", "Y_M", "HEADING_DEG"),
                   help="camera start pose in the world frame")
    args = p.parse_args()
    truth = {r["id"]: (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(args.truth))}
    sx, sy, sh = args.start[0], args.start[1], math.radians(args.start[2])

    rclpy.init()
    node = Align()
    print("Placement check. Press Ctrl+C to stop.", flush=True)
    try:
        while rclpy.ok():
            node.views.clear()
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            lines, turns = [], []
            for tag, obs in sorted(node.views.items()):
                if tag not in truth:
                    continue
                fwd, left = np.median(np.array(obs), axis=0)
                dx, dy = truth[tag][0] - sx, truth[tag][1] - sy
                exp_range = math.hypot(dx, dy)
                exp_bearing = math.remainder(math.atan2(dy, dx) - sh, 2 * math.pi)
                bearing = math.atan2(left, fwd)
                rng = math.hypot(fwd, left)
                # + = the robot is turned to the left (CCW) of the start heading
                turned = math.degrees(exp_bearing - bearing)
                frontal = abs(math.degrees(bearing)) <= FRONTAL_DEG
                if frontal:
                    turns.append(turned)
                lines.append(
                    f"  tag {tag}: fwd={fwd:.2f} left={left:+.2f} | range {rng:.2f} m "
                    f"(expected {exp_range:.2f}, {rng - exp_range:+.2f}) | "
                    f"heading error {turned:+.1f} deg"
                    f"{'' if frontal else ' (off-centre, not used)'}")
            print("-" * 60)
            if not lines:
                print("  No known tag in view.", flush=True)
                continue
            print("\n".join(lines))
            if not turns:
                print("  No tag near the image centre. Heading not checked.", flush=True)
                continue
            turn = float(np.mean(turns))
            if abs(turn) <= OK_DEG:
                print(f"  HEADING OK ({turn:+.1f} deg)", flush=True)
            else:
                side = "RIGHT (clockwise)" if turn > 0 else "LEFT (counter-clockwise)"
                print(f"  TURN THE ROBOT {abs(turn):.1f} deg to the {side}", flush=True)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
