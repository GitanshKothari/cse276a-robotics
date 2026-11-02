"""Navigator.

Drives the robot waypoint by waypoint using AprilTag localization.

The pose comes from TagLocalizer (see vision.py). Motion commands go to
/motor_commands (see motors.py). Position error is logged to CSV so
plot_speed.py and plot_trajectory.py can draw the result.
"""

import math
import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster

from .config import (
    DRY_RUN, USE_AXIS_MARGIN,
    FORWARD_CMD, FORWARD_SPEED,
    LEFT_CMD, LEFT_90_S, RIGHT_CMD, RIGHT_90_S,
    STEP_DISTANCE, STOP_PAUSE, COMMAND_DT, GLIDE_CMD, RAMP_SECONDS,
    ALONG_TOLERANCE, LATERAL_TOLERANCE, HEADING_TOLERANCE, MAX_CORRECTION,
    MAX_WAIT_FOR_TAG,
    SPEED_LOG, POSE_LOG,
)
from .vision import PoseEstimator, TagLocalizer, wrap


# -------- small helpers --------

def split_long_turn(angle_deg):
    """A single pivot is capped at 90 degrees. 135 becomes [90, 45]."""
    if abs(angle_deg) <= 90.0:
        return [angle_deg]
    sign = 1 if angle_deg > 0 else -1
    return [sign * 90.0, angle_deg - sign * 90.0]


# -------- motion driver --------

class MotionDriver:
    """Thin wrapper around the /motor_commands publisher."""

    def __init__(self, node, dry_run=False):
        self.dry_run = dry_run
        self.pub = node.create_publisher(
            Float32MultiArray, "/motor_commands", 10
        )

    def send(self, left, right):
        if self.dry_run:
            return
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    def stop(self):
        self.send(0.0, 0.0)


# -------- navigator node --------

class Navigator(Node):

    def __init__(self):
        super().__init__("navigator")

        cfg_dir = Path(get_package_share_directory("robot")) / "config"
        with open(cfg_dir / "landmarks.yaml") as f:
            landmarks = yaml.safe_load(f)
        with open(cfg_dir / "waypoints.yaml") as f:
            wp_cfg = yaml.safe_load(f)

        estimator = PoseEstimator(
            landmarks["tags"], landmarks["camera_mount"]
        )
        tag_ids = sorted(int(tid) for tid in landmarks["tags"].keys())

        scale = wp_cfg.get("scale", 1.0)
        self.waypoints = [
            (x * scale, y * scale, theta)
            for x, y, theta in wp_cfg["waypoints"]
        ]

        self.motion = MotionDriver(self, DRY_RUN)
        self.locator = TagLocalizer(self, estimator, tag_ids)

        # Foxglove outputs
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.path_pub = self.create_publisher(PathMsg, "/robot_path", 10)
        self.tf = TransformBroadcaster(self)
        self.path = PathMsg()
        self.path.header.frame_id = "world"

        # State
        self.target = 1
        self.turn_queue = []
        self.state = "LOCALIZE"
        self.state_start = time.monotonic()
        self.action_start = 0.0
        self.action_end = 0.0
        self.action_duration = 0.0
        self.action_left = 0.0
        self.action_right = 0.0
        self.ramp_from = 0.0
        self.last_motion_stamp = None
        self.was_forward = False

        # CSV logs
        self.speed_log = open(SPEED_LOG, "w")
        self.speed_log.write("t,left,right\n")
        self.pose_log = open(POSE_LOG, "w")
        self.pose_log.write("t,x,y,theta\n")
        self.t0 = time.monotonic()

        self.create_timer(COMMAND_DT, self.tick)
        self.get_logger().info(
            f"Navigator started | DRY_RUN={DRY_RUN} "
            f"| USE_AXIS_MARGIN={USE_AXIS_MARGIN}"
        )

    # ---------------- action helpers ----------------

    def _start_move(self, left, right, duration, description):
        self.action_left = left
        self.action_right = right
        self.action_start = time.monotonic()
        self.action_end = self.action_start + duration
        self.action_duration = duration
        self.was_forward = (
            left > 0 and right > 0 and abs(left - right) < 1e-6
        )
        self.ramp_from = GLIDE_CMD if self.was_forward else 0.0
        self.state = "MOVE"
        self.get_logger().info(description)

    def _ramped(self, now):
        """Trapezoid ramp for forward moves. Turns run at full command."""
        if not self.was_forward:
            return self.action_left, self.action_right
        t = now - self.action_start
        dur = self.action_duration
        tgt = max(abs(self.action_left), abs(self.action_right))
        start_scale = self.ramp_from / tgt if tgt > 0 else 1.0
        end_scale = GLIDE_CMD / tgt if tgt > 0 else 1.0
        ramp = min(RAMP_SECONDS, dur / 2.0)
        if t < ramp:
            s = start_scale + (1.0 - start_scale) * (t / ramp)
        elif t > dur - ramp:
            s = 1.0 + (end_scale - 1.0) * ((t - (dur - ramp)) / ramp)
        else:
            s = 1.0
        return self.action_left * s, self.action_right * s

    def _cruise_or_stop(self):
        """Keep gliding after a forward step. Stop fully after a turn."""
        if self.was_forward:
            self.motion.send(GLIDE_CMD, GLIDE_CMD)
        else:
            self.motion.stop()

    def _done(self, reason):
        self.state = "DONE"
        self.motion.stop()
        self.get_logger().info(f"STOPPED: {reason}")

    def _queue_turn(self, angle_rad, description):
        self.turn_queue = split_long_turn(math.degrees(angle_rad))
        self.get_logger().info(
            f"{description} | Turn sequence: {self.turn_queue}"
        )
        self._pop_turn()

    def _pop_turn(self):
        if not self.turn_queue:
            return
        angle_deg = self.turn_queue.pop(0)
        if angle_deg > 0:
            duration = LEFT_90_S * (angle_deg / 90.0)
            self._start_move(
                -LEFT_CMD, LEFT_CMD, duration,
                f"Turning LEFT {angle_deg:.1f} degrees",
            )
        else:
            duration = RIGHT_90_S * (abs(angle_deg) / 90.0)
            self._start_move(
                RIGHT_CMD, -RIGHT_CMD, duration,
                f"Turning RIGHT {abs(angle_deg):.1f} degrees",
            )

    # ---------------- Foxglove publish ----------------

    def _publish_pose(self, x, y, theta):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(theta / 2)
        msg.pose.orientation.w = math.cos(theta / 2)
        self.pose_pub.publish(msg)

        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(msg)
        self.path.poses = self.path.poses[-500:]
        self.path_pub.publish(self.path)

        tf = TransformStamped()
        tf.header = msg.header
        tf.child_frame_id = "robot_base"
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.rotation = msg.pose.orientation
        self.tf.sendTransform(tf)

    # ---------------- navigation decision ----------------

    def _navigate(self, x, y, theta):
        if self.target >= len(self.waypoints):
            self._done("All waypoints completed")
            return

        sx, sy, _ = self.waypoints[self.target - 1]
        gx, gy, _ = self.waypoints[self.target]

        dx, dy = gx - x, gy - y
        distance = math.hypot(dx, dy)

        vx, vy = gx - sx, gy - sy
        length = math.hypot(vx, vy)
        ux, uy = vx / length, vy / length
        rx, ry = x - sx, y - sy
        along = length - (rx * ux + ry * uy)
        cross = ux * ry - uy * rx

        at_pos = (
            abs(along) <= ALONG_TOLERANCE and abs(cross) <= LATERAL_TOLERANCE
            if USE_AXIS_MARGIN else distance <= ALONG_TOLERANCE
        )
        self.get_logger().info(
            f"WP={self.target} | Pose=({x:.2f}, {y:.2f}, "
            f"{math.degrees(theta):.1f} deg) | "
            f"Along={along:.2f} | Cross={cross:.2f}"
        )

        if at_pos:
            self._handle_reach()
            return

        # Heading correction (clipped).
        desired = self._desired_heading(x, y, dx, dy, vx, vy, along, cross)
        err = wrap(desired - theta)
        if abs(err) > HEADING_TOLERANCE:
            capped = max(-MAX_CORRECTION, min(MAX_CORRECTION, err))
            self._queue_turn(
                capped,
                f"Heading correction "
                f"{math.degrees(capped):.1f} deg "
                f"(raw {math.degrees(err):.1f})",
            )
            return

        # Forward step.
        remaining = (
            along if (USE_AXIS_MARGIN and along > ALONG_TOLERANCE)
            else distance
        )
        step = max(0.02, min(STEP_DISTANCE, remaining - ALONG_TOLERANCE))
        duration = step / FORWARD_SPEED
        self._start_move(
            FORWARD_CMD, FORWARD_CMD, duration,
            f"Forward {step:.2f} m for {duration:.2f} s",
        )

    def _desired_heading(self, x, y, dx, dy, vx, vy, along, cross):
        if not USE_AXIS_MARGIN:
            return math.atan2(dy, dx)
        heading = math.atan2(vy, vx)
        if abs(cross) > LATERAL_TOLERANCE:
            corr = max(
                -math.radians(30),
                min(math.radians(30), math.atan2(-cross, 0.5)),
            )
            heading = wrap(heading + corr)
        if along <= ALONG_TOLERANCE:
            heading = math.atan2(dy, dx)
        return heading

    def _handle_reach(self):
        previous = self.target
        self.target += 1
        self.get_logger().info(f"Reached waypoint {previous}")
        if self.target >= len(self.waypoints):
            # Final waypoint: align to its stored theta before stopping.
            self._align_final(previous)
            return
        # Turn from the heading of the last segment to the next segment.
        px, py, _ = self.waypoints[previous - 1]
        ax, ay, _ = self.waypoints[previous]
        nx, ny, _ = self.waypoints[self.target]
        prev_head = math.atan2(ay - py, ax - px)
        next_head = math.atan2(ny - ay, nx - ax)
        main_turn = wrap(next_head - prev_head)
        if abs(main_turn) > HEADING_TOLERANCE:
            self._queue_turn(
                main_turn, f"MAIN TURN {math.degrees(main_turn):.1f} deg"
            )

    def _align_final(self, final_idx):
        """Queue a turn so the robot ends at the final waypoint's theta."""
        px, py, _ = self.waypoints[final_idx - 1]
        fx, fy, final_theta = self.waypoints[final_idx]
        arrival_heading = math.atan2(fy - py, fx - px)
        turn = wrap(final_theta - arrival_heading)
        if abs(turn) <= HEADING_TOLERANCE:
            self._done("All waypoints completed")
            return
        self._queue_turn(
            turn, f"FINAL ALIGN {math.degrees(turn):.1f} deg "
                  f"to heading {math.degrees(final_theta):.1f} deg",
        )

    # ---------------- state machine ----------------

    def tick(self):
        now = time.monotonic()

        if self.state == "DONE":
            self.motion.stop()
            self._log_speed(now, 0.0, 0.0)
            return

        if self.state == "MOVE":
            if now < self.action_end:
                left, right = self._ramped(now)
                self.motion.send(left, right)
                self._log_speed(now, left, right)
                return
            self._cruise_or_stop()
            glide = GLIDE_CMD if self.was_forward else 0.0
            self._log_speed(now, glide, glide)
            self.last_motion_stamp = self.get_clock().now().nanoseconds
            if self.was_forward:
                self.locator.last_good_pose = None
            self.state = "PAUSE"
            self.state_start = now
            return

        self._cruise_or_stop()
        glide = GLIDE_CMD if self.was_forward else 0.0
        self._log_speed(now, glide, glide)

        if self.state == "PAUSE":
            if now - self.state_start >= STOP_PAUSE:
                if self.turn_queue:
                    self._pop_turn()
                elif self.target >= len(self.waypoints):
                    # Last waypoint reached and the alignment turn is done.
                    self._done("All waypoints completed")
                else:
                    self.state = "LOCALIZE"
                    self.state_start = now
            return

        # LOCALIZE
        if now - self.state_start > MAX_WAIT_FOR_TAG:
            self._done("No fresh AprilTag localization")
            return
        result = self.locator.get_pose(self.last_motion_stamp)
        if result is None:
            return
        x, y, theta, tag_id = result
        self._publish_pose(x, y, theta)
        self._log_pose(now, x, y, theta)
        self.get_logger().info(
            f"Tag {tag_id} | x={x:.2f}, y={y:.2f}, "
            f"theta={math.degrees(theta):.1f} deg"
        )
        self._navigate(x, y, theta)

    # ---------------- CSV logging ----------------

    def _log_speed(self, now, left, right):
        self.speed_log.write(
            f"{now - self.t0:.3f},{left:.3f},{right:.3f}\n"
        )

    def _log_pose(self, now, x, y, theta):
        self.pose_log.write(
            f"{now - self.t0:.3f},{x:.3f},{y:.3f},{theta:.4f}\n"
        )

    def destroy_node(self):
        self.motion.stop()
        for f in (self.speed_log, self.pose_log):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
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
