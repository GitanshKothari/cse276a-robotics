"""Record the run: a trajectory sample every half second, and the error
at each waypoint.

The assignment wants the plotted path to be what the algorithm believed,
not what it looked like from the side of the room, so the trajectory and
the waypoint estimates all come from the filter's own estimate.

If a ground truth is being published on ``true_pose`` - which only the
software-in-the-loop harness does, since the real floor has to be measured
by hand - it is recorded alongside in extra columns, and the waypoint rows
gain the true error next to the one the filter believed. On hardware those
columns stay empty and the hand-measured error goes in the report.
"""

import csv
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import String

from .transforms import wrap_angle, yaw_from_quaternion


class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        self.declare_parameter('output_dir', os.path.expanduser('~/hw2_runs'))
        self.declare_parameter('sample_period', 0.5)
        self.declare_parameter('run_name', '')
        # One "unit" for the distance report, e.g. the spacing between two
        # landmarks. The assignment asks for the total in these units.
        self.declare_parameter('distance_unit', 0.5)
        # Metres of wheel travel per radian turned on the spot, i.e. half
        # the track width. Used to fold turning into the total distance.
        self.declare_parameter('rotation_radius', 0.07)
        # Smallest step that counts as movement rather than estimator
        # jitter, so standing still does not accumulate distance.
        self.declare_parameter('translation_deadband', 0.002)
        self.declare_parameter('rotation_deadband_deg', 0.5)

        run_name = self.get_parameter('run_name').value or time.strftime('run_%Y%m%d_%H%M%S')
        self.directory = os.path.join(self.get_parameter('output_dir').value, run_name)
        os.makedirs(self.directory, exist_ok=True)

        self.trajectory_path = os.path.join(self.directory, 'trajectory.csv')
        self.waypoint_path = os.path.join(self.directory, 'waypoint_errors.csv')
        self.summary_path = os.path.join(self.directory, 'summary.json')

        self.trajectory_file = open(self.trajectory_path, 'w', newline='')
        self.trajectory_writer = csv.writer(self.trajectory_file)
        self.trajectory_writer.writerow(
            ['t', 'x', 'y', 'theta', 'position_std', 'heading_std',
             'tags_in_view', 'seconds_since_fix',
             'true_x', 'true_y', 'true_theta', 'estimate_error'])

        self.waypoint_file = open(self.waypoint_path, 'w', newline='')
        self.waypoint_writer = csv.writer(self.waypoint_file)
        self.waypoint_writer.writerow(
            ['index', 'target_x', 'target_y', 'target_theta',
             'estimate_x', 'estimate_y', 'estimate_theta',
             'position_error', 'heading_error_deg', 't',
             'true_x', 'true_y', 'true_theta',
             'true_position_error', 'true_heading_error_deg',
             'estimate_error'])

        self.translation_deadband = self.get_parameter('translation_deadband').value
        self.rotation_deadband = math.radians(
            self.get_parameter('rotation_deadband_deg').value)

        self.pose = None
        self.truth = None
        self.status = {}
        self.previous = None
        self.translation = 0.0
        self.rotation = 0.0
        self.samples = 0
        self.estimate_errors = []
        self.waypoint_truth = []
        self.start = None
        self.complete = False

        self.create_subscription(PoseWithCovarianceStamped, 'robot_pose', self.on_pose, 10)
        self.create_subscription(PoseStamped, 'true_pose', self.on_truth, 10)
        self.create_subscription(String, 'localization_status', self.on_status, 10)
        self.create_subscription(String, 'waypoint_status', self.on_waypoint, 10)
        self.create_timer(self.get_parameter('sample_period').value, self.on_sample)

        self.get_logger().info(f'Logging this run to {self.directory}')

    def on_pose(self, msg):
        orientation = msg.pose.pose.orientation
        self.pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
        )
        if self.start is None:
            self.start = time.monotonic()
        self.accumulate(self.pose)

    def accumulate(self, pose):
        """Add this step to the distance travelled.

        Accumulating here, at the rate the filter publishes, rather than off
        the half-second plot samples: summing chords across half-second gaps
        quietly under-reports anything but a straight line. The deadband is
        what stops the opposite error - the estimate jitters by a few
        millimetres while the rover sits still at a waypoint, and at 20 Hz
        that would otherwise pile up into distance the rover never drove.
        """
        if self.previous is None:
            self.previous = pose
            return

        step = math.hypot(pose[0] - self.previous[0], pose[1] - self.previous[1])
        turn = abs(wrap_angle(pose[2] - self.previous[2]))
        if step < self.translation_deadband and turn < self.rotation_deadband:
            return

        self.translation += step
        self.rotation += turn
        self.previous = pose

    def on_truth(self, msg):
        orientation = msg.pose.orientation
        self.truth = (
            msg.pose.position.x,
            msg.pose.position.y,
            yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
        )

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def on_sample(self):
        if self.pose is None or self.complete:
            return

        x, y, theta = self.pose
        elapsed = time.monotonic() - self.start

        if self.truth is None:
            truth_columns = ['', '', '', '']
        else:
            drift = math.hypot(x - self.truth[0], y - self.truth[1])
            self.estimate_errors.append(drift)
            truth_columns = [
                round(self.truth[0], 5), round(self.truth[1], 5),
                round(self.truth[2], 5), round(drift, 5),
            ]

        self.trajectory_writer.writerow([
            round(elapsed, 3), round(x, 5), round(y, 5), round(theta, 5),
            self.status.get('position_std', ''), self.status.get('heading_std', ''),
            self.status.get('tags_in_view', ''), self.status.get('seconds_since_fix', ''),
        ] + truth_columns)
        self.trajectory_file.flush()
        self.samples += 1

    def on_waypoint(self, msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return

        if event.get('event') == 'waypoint_reached':
            target = event['target']
            estimate = event['estimate']
            elapsed = 0.0 if self.start is None else time.monotonic() - self.start

            # The error the filter believed is the distance from the goal to
            # where it thought it was. The error that matters is from the
            # goal to where the rover actually ended up, and the gap between
            # the two is how much the localizer itself was wrong by.
            if self.truth is None:
                truth_columns = ['', '', '', '', '', '']
            else:
                true_x, true_y, true_theta = self.truth
                true_position = math.hypot(true_x - target[0], true_y - target[1])
                true_heading = wrap_angle(true_theta - target[2])
                drift = math.hypot(estimate[0] - true_x, estimate[1] - true_y)
                self.waypoint_truth.append((event['index'], true_position,
                                            math.degrees(true_heading), drift))
                truth_columns = [
                    round(true_x, 5), round(true_y, 5), round(true_theta, 5),
                    round(true_position, 5), round(math.degrees(true_heading), 2),
                    round(drift, 5),
                ]

            self.waypoint_writer.writerow([
                event['index'], target[0], target[1], target[2],
                estimate[0], estimate[1], estimate[2],
                event['position_error'],
                round(math.degrees(event['heading_error']), 2),
                round(elapsed, 2),
            ] + truth_columns)
            self.waypoint_file.flush()

        elif event.get('event') == 'route_complete':
            self.write_summary()

    def write_summary(self):
        if self.complete:
            return
        self.complete = True

        unit = self.get_parameter('distance_unit').value
        radius = self.get_parameter('rotation_radius').value
        turning_distance = self.rotation * radius
        total = self.translation + turning_distance

        summary = {
            'samples': self.samples,
            'duration_s': round(0.0 if self.start is None else time.monotonic() - self.start, 2),
            'translation_m': round(self.translation, 4),
            'rotation_rad': round(self.rotation, 4),
            'rotation_deg': round(math.degrees(self.rotation), 2),
            'turning_distance_m': round(turning_distance, 4),
            'total_distance_m': round(total, 4),
            'distance_unit_m': unit,
            'total_distance_units': round(total / unit, 3) if unit else None,
            'translation_units': round(self.translation / unit, 3) if unit else None,
        }

        if self.estimate_errors:
            summary['localization_error_mean_m'] = round(
                float(sum(self.estimate_errors) / len(self.estimate_errors)), 4)
            summary['localization_error_max_m'] = round(
                float(max(self.estimate_errors)), 4)
        if self.waypoint_truth:
            summary['waypoints'] = [
                {'index': index,
                 'true_position_error_m': round(position, 4),
                 'true_heading_error_deg': round(heading, 2),
                 'localization_error_m': round(drift, 4)}
                for index, position, heading, drift in self.waypoint_truth
            ]
        with open(self.summary_path, 'w') as handle:
            json.dump(summary, handle, indent=2)

        self.get_logger().info(
            f'Run complete: {summary["translation_m"]:.3f} m driven, '
            f'{summary["rotation_deg"]:.1f} deg turned, '
            f'total {summary["total_distance_m"]:.3f} m '
            f'= {summary["total_distance_units"]} units of {unit} m')
        self.get_logger().info(f'Wrote {self.summary_path}')

    def destroy_node(self):
        self.write_summary()
        for handle in (self.trajectory_file, self.waypoint_file):
            try:
                handle.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
