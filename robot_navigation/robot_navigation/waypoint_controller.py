"""Closed-loop waypoint controller.

HW1 drove the route open loop: the plan was turned into a list of timed
wheel commands and replayed blind. Here the same route is a list of goals,
and every control tick recomputes what to do from the current estimate, so
a wheel that slips or a turn that overshoots is corrected on the next tick
instead of being carried to the end of the run.

Each leg runs through three phases. The rover first turns on the spot
until it faces the goal, then drives towards it while steering, then turns
on the spot again to finish on the heading the waypoint asks for. Keeping
the large turns separate from the driving stops the rover from sweeping
out a wide arc when a leg starts with a big heading error.
"""

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from .kinematics import MAX_ANGULAR_LEFT, MAX_ANGULAR_RIGHT, MAX_LINEAR, twist_to_wheels
from .transforms import wrap_angle, yaw_from_quaternion

ALIGN = 'align'
DRIVE = 'drive'
SETTLE = 'settle'
SEARCH = 'search'
DONE = 'done'


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_controller')

        # Flat [x, y, theta, x, y, theta, ...]; the assignment's route.
        self.declare_parameter('waypoints', [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 2.0, math.pi,
            0.0, 0.0, 0.0,
        ])
        self.declare_parameter('control_rate', 20.0)

        self.declare_parameter('k_rho', 0.6)
        self.declare_parameter('k_alpha', 1.6)
        self.declare_parameter('k_beta', 1.4)

        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 0.9)
        self.declare_parameter('min_cmd', 0.10)
        # A spin scrubs both tyres sideways and needs far more than a
        # drive does. Measure yours: hold a spin command and raise it
        # until the rover reliably starts to turn.
        self.declare_parameter('min_turn_cmd', 0.35)

        self.declare_parameter('position_tolerance', 0.06)
        self.declare_parameter('heading_tolerance_deg', 6.0)
        self.declare_parameter('align_tolerance_deg', 12.0)
        self.declare_parameter('align_exit_deg', 45.0)
        self.declare_parameter('settle_time', 0.4)

        # What to do while no landmark is in view.
        self.declare_parameter('blind_slow_after', 1.5)
        self.declare_parameter('blind_stop_after', 5.0)
        self.declare_parameter('blind_speed_scale', 0.5)
        self.declare_parameter('search_angular', 0.5)
        self.declare_parameter('search_sweep', 2.5)
        self.declare_parameter('pose_timeout', 0.5)

        self.declare_parameter('start_delay', 2.0)

        # Seconds to hold the rover stopped after the last waypoint before
        # shutting the node down; 0 keeps it alive holding still, which is
        # what you want on the hardware so the stack stays up for
        # inspection. The simulated sweeps set it so a scripted run ends.
        self.declare_parameter('shutdown_after_complete', 0.0)

        values = list(self.get_parameter('waypoints').value)
        if len(values) < 6 or len(values) % 3:
            raise RuntimeError('waypoints must be a flat list of x, y, theta triples')
        self.waypoints = [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]

        self.pose = None
        self.pose_stamp = 0.0
        self.seconds_since_fix = None
        self.index = 1          # waypoint 0 is where the rover starts
        self.phase = ALIGN
        self.phase_entered = time.monotonic()
        self.started = time.monotonic()
        self.search_started = 0.0
        self.search_direction = 1.0
        self.resume_phase = ALIGN

        self.command_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.status_pub = self.create_publisher(String, 'waypoint_status', 10)
        self.create_subscription(PoseWithCovarianceStamped, 'robot_pose', self.on_pose, 10)
        self.create_subscription(String, 'localization_status', self.on_localization, 10)

        period = 1.0 / max(self.get_parameter('control_rate').value, 1.0)
        self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f'Driving {len(self.waypoints) - 1} legs: '
            + ' -> '.join(f'({x:.2f}, {y:.2f}, {t:.2f})' for x, y, t in self.waypoints))

    # ------------------------------------------------------------------

    def on_pose(self, msg):
        orientation = msg.pose.pose.orientation
        self.pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
        )
        self.pose_stamp = time.monotonic()

    def on_localization(self, msg):
        try:
            self.seconds_since_fix = json.loads(msg.data).get('seconds_since_fix')
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------

    def on_timer(self):
        now = time.monotonic()

        if self.phase == DONE:
            self.publish(0.0, 0.0)
            linger = self.get_parameter('shutdown_after_complete').value
            if linger > 0.0 and now - self.phase_entered > linger:
                self.get_logger().info('Route complete, shutting down')
                raise SystemExit(0)
            return

        # Give the localizer a moment to see a tag before moving at all.
        if now - self.started < self.get_parameter('start_delay').value:
            self.publish(0.0, 0.0)
            return

        if self.pose is None or now - self.pose_stamp > self.get_parameter('pose_timeout').value:
            self.get_logger().warn('No pose estimate, holding still', throttle_duration_sec=2.0)
            self.publish(0.0, 0.0)
            return

        blind = self.seconds_since_fix
        if self.phase != SEARCH and blind is not None:
            if blind > self.get_parameter('blind_stop_after').value:
                self.enter_search(now)

        if self.phase == SEARCH:
            self.run_search(now)
            return

        self.run_leg(now, blind)

    def run_leg(self, now, blind):
        x, y, theta = self.pose
        goal_x, goal_y, goal_theta = self.waypoints[self.index]

        dx, dy = goal_x - x, goal_y - y
        rho = math.hypot(dx, dy)
        alpha = wrap_angle(math.atan2(dy, dx) - theta)
        beta = wrap_angle(goal_theta - theta)

        position_tolerance = self.get_parameter('position_tolerance').value
        heading_tolerance = math.radians(self.get_parameter('heading_tolerance_deg').value)
        align_tolerance = math.radians(self.get_parameter('align_tolerance_deg').value)
        align_exit = math.radians(self.get_parameter('align_exit_deg').value)

        max_linear = min(self.get_parameter('max_linear').value, MAX_LINEAR)
        max_angular = self.get_parameter('max_angular').value

        # Slow down rather than stop when the tags have been gone a little
        # while: the pose is drifting on dead reckoning, and less distance
        # covered blind means less drift to correct once a tag reappears.
        if blind is not None and blind > self.get_parameter('blind_slow_after').value:
            scale = self.get_parameter('blind_speed_scale').value
            max_linear *= scale
            max_angular *= scale

        if self.phase == ALIGN:
            if rho < position_tolerance or abs(alpha) < align_tolerance:
                self.set_phase(DRIVE, now)
            else:
                self.drive(0.0, clamp(self.get_parameter('k_alpha').value * alpha, max_angular))
                return

        if self.phase == DRIVE:
            if rho < position_tolerance:
                self.set_phase(SETTLE, now)
            else:
                if abs(alpha) > align_exit:
                    # Drifted far enough off that steering while driving
                    # would carve a long arc; turn on the spot instead.
                    self.set_phase(ALIGN, now)
                    return
                v = clamp(self.get_parameter('k_rho').value * rho, max_linear)
                omega = clamp(self.get_parameter('k_alpha').value * alpha, max_angular)
                # Tight turns first: giving up forward speed when the
                # heading error is large keeps the path from bulging out.
                v *= max(0.0, math.cos(alpha))
                self.drive(v, omega)
                return

        if self.phase == SETTLE:
            if abs(beta) < heading_tolerance:
                if now - self.phase_entered >= self.get_parameter('settle_time').value:
                    self.reach_waypoint(rho, beta)
                else:
                    self.publish(0.0, 0.0)
            else:
                self.drive(0.0, clamp(self.get_parameter('k_beta').value * beta, max_angular))

    def run_search(self, now):
        """Turn on the spot to bring a landmark back into view."""
        blind = self.seconds_since_fix
        if blind is not None and blind < self.get_parameter('blind_stop_after').value:
            self.get_logger().info('Landmark reacquired, resuming')
            self.set_phase(self.resume_phase, now)
            return

        sweep = self.get_parameter('search_sweep').value
        if now - self.search_started > sweep:
            # Sweep one way, then twice as far back the other way, so the
            # scan widens around wherever the rover was looking.
            self.search_direction *= -1.0
            self.search_started = now
        self.drive(0.0, self.search_direction * self.get_parameter('search_angular').value)

    def reach_waypoint(self, rho, beta):
        x, y, theta = self.pose
        goal = self.waypoints[self.index]
        self.publish(0.0, 0.0)
        self.get_logger().info(
            f'Waypoint {self.index} reached: estimate ({x:.3f}, {y:.3f}, {theta:.3f}) '
            f'target ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) '
            f'error {rho:.3f} m, {math.degrees(beta):.1f} deg')
        self.status_pub.publish(String(data=json.dumps({
            'event': 'waypoint_reached',
            'index': self.index,
            'target': list(goal),
            'estimate': [round(x, 4), round(y, 4), round(theta, 4)],
            'position_error': round(rho, 4),
            'heading_error': round(beta, 4),
        })))

        self.index += 1
        if self.index >= len(self.waypoints):
            self.set_phase(DONE, time.monotonic())
            self.get_logger().info('Route complete')
            self.status_pub.publish(String(data=json.dumps({'event': 'route_complete'})))
        else:
            self.set_phase(ALIGN, time.monotonic())

    # ------------------------------------------------------------------

    def enter_search(self, now):
        self.get_logger().warn('No landmark for too long, stopping to look around')
        self.resume_phase = self.phase
        self.search_started = now
        self.search_direction = 1.0
        self.set_phase(SEARCH, now)
        self.status_pub.publish(String(data=json.dumps({'event': 'search_started'})))

    def set_phase(self, phase, now):
        if phase != self.phase:
            self.phase = phase
            self.phase_entered = now

    def drive(self, v, omega):
        # Never ask for a twist the calibration cannot deliver.
        limit = MAX_ANGULAR_LEFT if omega >= 0.0 else MAX_ANGULAR_RIGHT
        omega = clamp(omega, limit)
        left, right = twist_to_wheels(
            v, omega,
            self.get_parameter('min_cmd').value,
            self.get_parameter('min_turn_cmd').value)
        self.publish(left, right)

    def publish(self, left, right):
        self.command_pub.publish(Float32MultiArray(data=[float(left), float(right)]))


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Leave the rover stopped rather than relying on the watchdog.
        for _ in range(3):
            node.publish(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
