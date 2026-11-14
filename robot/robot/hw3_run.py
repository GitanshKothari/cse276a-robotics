import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

from .config import (
    CAMERA_FRAME, COMMAND_DT, FORWARD_CMD,
    LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S, TAG_FRAME_PREFIX,
)
from .hw3_slam import Slam, rotation


# Run settings
SIDES = 4                      # Number of polygon sides (4 = square, 8 = octagon).
SIDE_M = 0.5                   # Length of each side in metres.
LAPS = 1                      # Number of times to complete the path.
OUTPUT_DIR = Path.home() / "hw3_results"  # Folder for saved maps and paths.

# Movement calibration
STEP_DISTANCE_M = 0.20         # Distance driven between AprilTag observations.
STEP_SPEED_MPS = 0.233         # Measured speed for short forward/reverse steps.
TURN_DELAY_S = 0.025          # Motor startup delay for each turn pulse.
MAX_TURN_STEP_DEG = 25        # Maximum angle commanded in one turn pulse.
TURN_REFERENCE_DEG = 25       # Angle used for the turn-pulse speed calibration.
TURN_TOLERANCE_DEG = 4        # Allowed heading error when first aiming at a goal.
DRIVE_TOLERANCE_DEG = 6       # Allowed heading error while moving along a side.
GOAL_TOLERANCE_M = 0.04       # Distance considered close enough to a waypoint.
NO_TURN_DISTANCE_M = 0.30     # Avoid turning when within this range of a waypoint.
STEP_LOOK_TIME_S = 0.6        # Time spent detecting tags after each move.
CAMERA_YAW_CORRECTION_RAD = 0.059  # Measured camera angle offset from robot heading.

# AprilTag filtering
MAX_TAG_ERROR_M = 0.55        # Reject a known tag if its position differs too much from the map.
NEW_TAG_MAX_SIDE_RATIO = 0.50     # Maximum |left/forward| for a new tag.
KNOWN_TAG_MAX_SIDE_RATIO = 0.80   # Maximum |left/forward| for a known tag.
MAP_UPDATE_MAX_SIDE_RATIO = 0.60  # Maximum |left/forward| to update a tag's map position.
NEW_TAG_MAX_FORWARD_M = 2.20     # Maximum forward range for adding a new tag.
KNOWN_TAG_MAX_FORWARD_M = 2.30   # Maximum forward range for observing a known tag.
MAP_UPDATE_MAX_FORWARD_M = 2.30  # Maximum forward range to update a tag's map position.
NEW_TAG_MIN_READINGS = 4         # Different camera readings required before adding a tag.

# Other thresholds
MIN_TAG_FORWARD_M = 0.20     # Ignore AprilTags closer than this to the camera.
MAX_TAG_FORWARD_M = 3.2      # Ignore AprilTags farther than this along the camera axis.
STOP_WAIT_S = 0.30           # Pause for the robot to settle before reading tags.
DEFAULT_LOOK_TIME_S = 1.5   # Normal observation time after a turn or at a corner.
MIN_MOVE_M = 0.02           # Ignore movements shorter than this distance.
AWAY_FROM_GOAL_M = 0.08     # Count a step as worse if distance to goal increases by this much.
NEAR_GOAL_TURN_M = 0.35     # Use the tighter heading tolerance when this close to a goal.
RETURN_NO_TURN_M = 0.08     # Smaller no-turn zone when correcting the final position.


def corners(sides, side_m):
    x = y = 0.0
    for i in range(sides):
        angle = i * 2 * math.pi / sides
        x += side_m * math.cos(angle)
        y += side_m * math.sin(angle)
        yield x, y


class Run(Node):
    def __init__(self):
        super().__init__("hw3_run")
        self.motor = self.create_publisher(Float32MultiArray, "/motor_commands", 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 10)

        self.slam = Slam()
        self.path = []
        self.seen = set()
        self.since = 0
        self.pending = {}
        self.pending_stamps = {}

    def cmd(self, left, right):
        self.motor.publish(Float32MultiArray(data=[left, right]))

    def on_tf(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id != CAMERA_FRAME:
                continue

            name = tf.child_frame_id
            if not name.startswith(TAG_FRAME_PREFIX):
                continue

            tag = name[len(TAG_FRAME_PREFIX):]
            stamp = Time.from_msg(tf.header.stamp).nanoseconds
            if not tag.isdigit() or tag in self.seen or stamp < self.since:
                continue

            p = tf.transform.translation
            if not (MIN_TAG_FORWARD_M < p.z < MAX_TAG_FORWARD_M and math.isfinite(p.z) and math.isfinite(p.x)):
                continue

            forward, left = float(p.z), float(-p.x)
            c = math.cos(CAMERA_YAW_CORRECTION_RAD)
            s = math.sin(CAMERA_YAW_CORRECTION_RAD)
            forward, left = forward * c - left * s, left * c + forward * s
            side_ratio = abs(left) / forward

            if tag in self.slam.index:
                if side_ratio > KNOWN_TAG_MAX_SIDE_RATIO or forward > KNOWN_TAG_MAX_FORWARD_M:
                    continue

                i = self.slam.index[tag]
                expected = rotation(self.slam.x[2]).T @ (
                    self.slam.x[i:i + 2] - self.slam.x[:2]
                )
                if math.hypot(forward - expected[0], left - expected[1]) > MAX_TAG_ERROR_M:
                    continue

                joint = side_ratio <= MAP_UPDATE_MAX_SIDE_RATIO and forward <= MAP_UPDATE_MAX_FORWARD_M
                self.slam.observe(tag, forward, left, update_landmark=joint)
                self.seen.add(tag)
                continue

            if side_ratio > NEW_TAG_MAX_SIDE_RATIO or forward > NEW_TAG_MAX_FORWARD_M:
                continue
            if self.pending_stamps.get(tag) == stamp:
                continue

            self.pending_stamps[tag] = stamp
            observations = self.pending.setdefault(tag, [])
            observations.append((forward, left))
            if len(observations) < NEW_TAG_MIN_READINGS:
                continue

            median_forward = float(np.median([v[0] for v in observations]))
            median_left = float(np.median([v[1] for v in observations]))
            self.slam.observe(tag, median_forward, median_left)

            self.seen.add(tag)
            del self.pending[tag]
            self.pending_stamps.pop(tag, None)

    def look(self, seconds=DEFAULT_LOOK_TIME_S):
        self.cmd(0.0, 0.0)
        time.sleep(STOP_WAIT_S)

        self.pending.clear()
        self.pending_stamps.clear()
        self.since = self.get_clock().now().nanoseconds
        self.seen.clear()

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

        camera_pose = self.slam.x[:3].copy()
        axle_xy = camera_pose[:2] - rotation(camera_pose[2]) @ self.slam.camera_offset
        self.path.append([float(axle_xy[0]), float(axle_xy[1]), float(camera_pose[2])])

    def move(self, left, right, duration):
        start = time.monotonic()
        while time.monotonic() - start < duration:
            self.cmd(left, right)
            time.sleep(min(COMMAND_DT, max(0.0, duration - (time.monotonic() - start))))
        self.cmd(0.0, 0.0)
        return time.monotonic() - start

    def turn_to(self, heading):
        error = math.remainder(heading - self.slam.x[2], 2 * math.pi)
        if abs(error) <= math.radians(TURN_TOLERANCE_DEG):
            return True

        if error > 0:
            seconds_90 = LEFT_90_S
            wheels = (-LEFT_CMD, LEFT_CMD)
        else:
            seconds_90 = RIGHT_90_S
            wheels = (RIGHT_CMD, -RIGHT_CMD)

        cal = math.radians(TURN_REFERENCE_DEG)
        rate = cal / (cal / math.radians(90) * seconds_90 - TURN_DELAY_S)
        turn = math.copysign(min(abs(error), math.radians(MAX_TURN_STEP_DEG)), error)

        dt = self.move(*wheels, TURN_DELAY_S + abs(turn) / rate)
        self.slam.predict(0.0, math.copysign(rate * max(0.0, dt - TURN_DELAY_S), turn))
        self.look()
        return False

    def forward(self, distance):
        if abs(distance) <= MIN_MOVE_M:
            self.look()
            return

        wheel = math.copysign(FORWARD_CMD, distance)
        dt = self.move(wheel, wheel, abs(distance) / STEP_SPEED_MPS)
        self.slam.predict(math.copysign(STEP_SPEED_MPS * dt, distance), 0.0)
        self.look(seconds=STEP_LOOK_TIME_S)

    def drive_to(self, gx, gy, side_m, reverse_ok=False, no_turn_m=NO_TURN_DISTANCE_M):
        last_distance = None
        worse = 0

        for i in range(10 + 3 * math.ceil(side_m / STEP_DISTANCE_M)):
            camera_x, camera_y, heading = self.slam.x[:3]
            axle_xy = np.array([camera_x, camera_y]) - rotation(heading) @ self.slam.camera_offset
            x, y = axle_xy

            dx, dy = gx - x, gy - y
            distance = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)

            if distance <= GOAL_TOLERANCE_M:
                return

            if last_distance is not None and distance > last_distance + AWAY_FROM_GOAL_M:
                worse += 1
                if worse >= 3:
                    raise RuntimeError(f"moving away from ({gx:.2f},{gy:.2f})")
            else:
                worse = 0
            last_distance = distance

            if distance < no_turn_m:
                ahead = dx * math.cos(heading) + dy * math.sin(heading)
                if ahead > GOAL_TOLERANCE_M:
                    self.forward(min(ahead, STEP_DISTANCE_M))
                    continue
                if reverse_ok and ahead < -GOAL_TOLERANCE_M:
                    self.forward(max(ahead, -STEP_DISTANCE_M))
                    continue
                return

            tolerance = TURN_TOLERANCE_DEG if (i == 0 or distance < NEAR_GOAL_TURN_M) else DRIVE_TOLERANCE_DEG
            if abs(math.remainder(bearing - heading, 2 * math.pi)) > math.radians(tolerance):
                if not self.turn_to(bearing):
                    continue

            self.forward(min(distance, STEP_DISTANCE_M))

        raise RuntimeError(f"could not reach axle endpoint ({gx:.2f},{gy:.2f})")

    def drive(self, sides, side_m, laps, out):
        start = time.monotonic()
        while self.motor.get_subscription_count() == 0:
            if time.monotonic() - start > 5:
                raise RuntimeError("no motor subscriber; start hw3.launch.py")
            time.sleep(0.1)

        self.look()
        self.look()
        if not self.slam.index:
            print("WARNING: no AprilTags detected at start")

        initial_camera = self.slam.x[:3].copy()
        initial_heading = float(initial_camera[2])
        R0 = rotation(initial_heading)
        axle_start = initial_camera[:2] - R0 @ self.slam.camera_offset
        goals = [axle_start + R0 @ np.array([gx, gy]) for gx, gy in corners(sides, side_m)]

        for lap in range(1, laps + 1):
            for gx, gy in goals:
                self.drive_to(gx, gy, side_m)
                self.look()

            for _ in range(6):
                if self.turn_to(initial_heading):
                    break
            self.look()

            try:
                self.drive_to(*goals[-1], side_m, reverse_ok=True, no_turn_m=RETURN_NO_TURN_M)
            except RuntimeError as error:
                print(f"WARNING: {error}")

            for _ in range(6):
                if self.turn_to(initial_heading):
                    break
            self.look()

            data = {
                "lap": lap,
                "sides": sides,
                "side_m": side_m,
                "path": self.path,
                "path_reference": "axle",
                "camera_offset": self.slam.camera_offset.tolist(),
                "tags": self.slam.landmarks(),
            }
            (out / f"lap-{lap}.json").write_text(json.dumps(data, indent=2))


def main():
    out = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)

    rclpy.init(args=[])
    run = Run()
    try:
        run.drive(SIDES, SIDE_M, LAPS, out)
    finally:
        if rclpy.ok():
            run.cmd(0.0, 0.0)
        run.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
