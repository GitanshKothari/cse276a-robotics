"""Waypoint navigation using AprilTag localization."""

import math
import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .config import (
    FORWARD_CMD, FORWARD_SPEED, LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S, STEP_DISTANCE, STOP_PAUSE, COMMAND_DT, GLIDE_CMD, RAMP_SECONDS, ALONG_TOLERANCE, LATERAL_TOLERANCE, HEADING_TOLERANCE, MAX_CORRECTION, MAX_WAIT_FOR_TAG,
)
from .vision import PoseEstimator, TagLocalizer, wrap


class Navigator(Node):
    def __init__(self):
        super().__init__("navigator")

        config = Path(get_package_share_directory("robot")) / "config"
        
        #loading waypoints and landmarks 
        with open(config / "landmarks.yaml") as f:
            landmarks = yaml.safe_load(f)
        with open(config / "waypoints.yaml") as f:
            data = yaml.safe_load(f)

        # making sure the waypoints are scaled correctly. 
        scale = data.get("scale", 1.0)
        self.waypoints = [(x * scale, y * scale, theta)
                          for x, y, theta in data["waypoints"]]
        
        
        # only use the lowesst tag number. 
        tag_ids = sorted(int(t) for t in landmarks["tags"])
        estimator = PoseEstimator(landmarks["tags"], landmarks["camera_mount"])
        self.locator = TagLocalizer(self, estimator, tag_ids)

        self.motor_pub = self.create_publisher(Float32MultiArray, "/motor_commands", 10)

        self.target = 1
        self.state = "LOCALIZE"
        self.state_start = time.monotonic()
        self.turns = []
        self.last_motion_stamp = None
        self.was_forward = False
        self.move_start = 0.0
        self.move_end = 0.0
        self.move_duration = 0.0
        self.left = 0.0
        self.right = 0.0

        self.create_timer(COMMAND_DT, self.tick)

    def send_motors(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.motor_pub.publish(msg)

    def stop(self, reason):
        self.state = "DONE"
        self.send_motors(0, 0)

    def start_move(self, left, right, duration):
        # logging stuff before starting a move

        self.left, self.right = left, right
        self.move_start = time.monotonic()
        self.move_duration = duration
        self.move_end = self.move_start + duration
        self.was_forward = left > 0 and right > 0 and abs(left - right) < 1e-6
        self.state = "MOVE"

    def motor_speeds(self, now):
        
        # this is to ensure the robot is moving a bit more smoothly
        if not self.was_forward:
            return self.left, self.right

        elapsed_time = now - self.move_start
        ramp = min(RAMP_SECONDS, self.move_duration / 2)
        command = max(abs(self.left), abs(self.right))
        glide_scale = GLIDE_CMD / command if command else 1.0

        if ramp > 0 and elapsed_time < ramp:
            scale = glide_scale + (1 - glide_scale) * elapsed_time / ramp
        elif ramp > 0 and elapsed_time > self.move_duration - ramp:
            scale = 1 + (glide_scale - 1) * (elapsed_time - self.move_duration + ramp) / ramp
        else:
            scale = 1.0
        return self.left * scale, self.right * scale

    def cruise_or_stop(self):
        # stops if the previous command was turn else glides for forward command
        speed = GLIDE_CMD if self.was_forward else 0.0
        self.send_motors(speed, speed)
        return speed

    def start_turn(self, angle):
        # split the turn commands to be below 90 deg at any point
        degrees = math.degrees(angle)
        sign = 1 if degrees > 0 else -1
        self.turns = []
        while abs(degrees) > 90:
            self.turns.append(sign * 90.0)
            degrees -= sign * 90.0
        if abs(degrees) > 1e-9:
            self.turns.append(degrees)
        self.next_turn()

    def next_turn(self):
        if not self.turns:
            return
        angle = self.turns.pop(0)
        if angle > 0:
            self.start_move(-LEFT_CMD, LEFT_CMD, LEFT_90_S * angle / 90)
        else:
            self.start_move(RIGHT_CMD, -RIGHT_CMD, RIGHT_90_S * abs(angle) / 90)


 
    def navigate(self, x, y, theta):
        next_x, next_y, _ = self.waypoints[self.target]
        dx, dy = next_x - x, next_y - y
        distance = math.hypot(dx, dy)

        if distance <= ALONG_TOLERANCE:
            self.reached_waypoint()
            return

        desired = math.atan2(dy, dx)
        error = wrap(desired - theta)

        if abs(error) > HEADING_TOLERANCE:
            self.start_turn(max(-MAX_CORRECTION, min(MAX_CORRECTION, error)))
            return

        step = max(0.02, min(STEP_DISTANCE, distance - ALONG_TOLERANCE))
        self.start_move(FORWARD_CMD, FORWARD_CMD, step / FORWARD_SPEED)


    def reached_waypoint(self):
        current = self.target
        self.target += 1

        if self.target >= len(self.waypoints):
            previous_X, previous_y, _ = self.waypoints[current - 1]
            x, y, final_theta = self.waypoints[current]

            arrival_heading = math.atan2(y - previous_y, x - previous_X)


            angle = wrap(final_theta - arrival_heading)
            if abs(angle) > HEADING_TOLERANCE:
                self.start_turn(angle)
            else:
                self.stop("All waypoints completed")
            return

        previous_X, previous_y, _ = self.waypoints[current - 1]
        x, y, _ = self.waypoints[current]
        next_x, next_y, _ = self.waypoints[self.target]
        previous_heading = math.atan2(y - previous_y, x - previous_X)
        next_heading = math.atan2(next_y - y, next_x - x)
        
        angle = wrap(next_heading - previous_heading)
        if abs(angle) > HEADING_TOLERANCE:
            self.start_turn(angle)

    def tick(self):
        now = time.monotonic()
        if self.state == "DONE":
            self.send_motors(0, 0)
            return

        if self.state == "MOVE":
            if now < self.move_end:
                left, right = self.motor_speeds(now)
                self.send_motors(left, right)
                return
            self.cruise_or_stop()
            self.last_motion_stamp = self.get_clock().now().nanoseconds
            if self.was_forward:
                self.locator.last_good_pose = None
            self.state = "PAUSE"
            self.state_start = now
            return

        self.cruise_or_stop()

        if self.state == "PAUSE":
            if now - self.state_start >= STOP_PAUSE:
                if self.turns:
                    self.next_turn()
                elif self.target >= len(self.waypoints):
                    self.stop("All waypoints completed")
                else:
                    self.state = "LOCALIZE"
                    self.state_start = now
            return

        if now - self.state_start > MAX_WAIT_FOR_TAG:
            self.stop("No fresh AprilTag localization")
            return

        result = self.locator.get_pose(self.last_motion_stamp)
        if result is None:
            return

        x, y, theta, _ = result
        self.navigate(x, y, theta)

    def destroy_node(self):
        self.send_motors(0, 0)
        super().destroy_node()


def main():
    rclpy.init()
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
