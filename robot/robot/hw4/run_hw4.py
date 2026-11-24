#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

import arena
from planning import clearance, plan, plot
from slam import Slam, rotation, wrap

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

ROUTES = ("distance", "safety", "prm")
HERE = Path(__file__).resolve().parent


def detector(name, family, size):
    return ["ros2", "run", "apriltag_ros", "apriltag_node", "--ros-args",
            "-r", f"__node:={name}", "-r", f"detections:=/{name}/detections",
            "-p", f"family:={family}", "-p", f"size:={float(size)}",
            "-p", "image_transport:=compressed", "-p", "max_hamming:=1",
            "-p", "detector.threads:=2", "-p", "detector.decimate:=2.0",
            "-p", "pose_estimation_method:=pnp",
            "-r", "image_rect:=/camera/image_raw", "-r", "camera_info:=/camera/camera_info"]


def tag_key(frame, mode):
    family, _, number = frame.partition(":")
    if not number.isdigit():
        return None
    tag = int(number)
    if family == f"tag{arena.BOUNDARY_TAG_FAMILY}" and tag in arena.BOUNDARY_TAGS:
        return tag
    if (mode == "survey" and family == f"tag{arena.OBSTACLE_TAG_FAMILY}"
            and tag in arena.OBSTACLE_TAGS):
        return frame
    return None


@contextmanager
def hardware(output, include_obstacle):
    commands = [
        ("camera", ["ros2", "launch", "robot_vision_camera", "robot_vision_camera.launch.py",
                    "image_rectify:=true"]),
        ("boundary_tags", detector("hw4_boundary_tags", arena.BOUNDARY_TAG_FAMILY, arena.TAG_SIZE)),
        ("motor", [sys.executable, "-m", "robot.motor_control"]),
    ]
    if include_obstacle:
        commands.append(("obstacle_tags", detector("hw4_obstacle_tags", arena.OBSTACLE_TAG_FAMILY,
                                                   arena.OBSTACLE_TAG_SIZE)))
    children = {}
    env = os.environ.copy()
    extra = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(HERE.parent.parent) + (os.pathsep + extra if extra else "")
    try:
        for key, command in commands:
            with (output / f"{key}.log").open("w") as log:
                children[key] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, env=env)
        yield children
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            for process in children.values():
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(0.3 if sig == signal.SIGINT else 0.1)
        for process in children.values():
            process.wait()


def save_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def sightings_plot(file, mode, sightings, trace, cylinder):
    """SVG of where each tag reading placed its tag (green used, red rejected)."""
    xmin, ymin, xmax, ymax = arena.BOUNDS
    scale = 520 / max(xmax - xmin, ymax - ymin)

    def screen(p):
        return 50 + scale * (p[0] - xmin), 55 + scale * (ymax - p[1])

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="665" font-family="sans-serif">',
           '<rect width="650" height="665" fill="white"/>',
           f'<text x="50" y="32" font-size="20">HW4 {mode}: where the robot saw each tag</text>',
           f'<rect x="50" y="55" width="{(xmax - xmin) * scale}" height="{(ymax - ymin) * scale}" '
           'fill="#f8fafc" stroke="#334155"/>']
    cx, cy = screen(cylinder)
    svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cylinder[2] * scale:.1f}" fill="#cbd5e1"/>')
    if len(trace) > 1:
        points = " ".join("%.1f,%.1f" % screen(p) for p in trace)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.5" opacity="0.6"/>')
    for s in sightings:
        x, y = screen(s["implied_tag_xy"])
        rx, ry = screen(s["robot_pose"])
        color = "#16a34a" if s["accepted"] else "#dc2626"
        svg.append(f'<line x1="{rx:.1f}" y1="{ry:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" '
                   'stroke-width="0.6" opacity="0.25"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" opacity="0.7"/>')
        label = s["tag"].replace("tagStandard41h12:", "C")
        svg.append(f'<text x="{x + 4:.1f}" y="{y + 10:.1f}" font-size="9" fill="{color}">{label}</text>')
    for tag, p in arena.BOUNDARY_TAGS.items():
        x, y = screen(p)
        svg.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="none" '
                   'stroke="#0f172a" stroke-width="2"/>')
        svg.append(f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="13" font-weight="bold">{tag}</text>')
    svg += ['<text x="50" y="605" font-size="12">Black squares: tag positions in arena.py. Dots: where each '
            'reading put the tag</text>',
            '<text x="50" y="622" font-size="12">(green = used, red = rejected; C1-C3 = cylinder tags). '
            'Faint lines join robot to sighting.</text>',
            '<text x="50" y="639" font-size="12">Blue: estimated robot path.</text>', '</svg>']
    Path(file).write_text("\n".join(svg))


class Robot(Node):
    """Localize on the boundary tags and drive a planned route in short pulses."""
    MAX_TAG_LAG_NS = 500_000_000

    def __init__(self, mode, cylinder, folder, children):
        super().__init__("hw4")
        self.mode, self.cylinder, self.folder, self.children = mode, cylinder, folder, children
        self.motor = self.create_publisher(Float32MultiArray, "/motor_commands", 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, qos_profile_sensor_data)
        self.filter = Slam()
        self.readings = {}
        self.after_ns = None
        self.last_anchor = -math.inf
        self.anchors = set()
        self.trace = []
        self.begin = time.monotonic()
        self.drive_started = None
        self.wheel_model_distance = 0.0
        self.route, self.nodes, self.edges = [], [], []
        self.mapped = None

    def command(self, left=0.0, right=0.0):
        self.motor.publish(Float32MultiArray(data=[float(left), float(right)]))

    def check(self):
        if not rclpy.ok() or time.monotonic() - self.begin > 720:
            raise RuntimeError("ROS stopped or the run timed out.")
        for name, process in self.children.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited. See {self.folder.parent / (name + '.log')}")

    def on_tf(self, message):
        if self.after_ns is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        for tf in message.transforms:
            if tf.header.frame_id != arena.CAMERA_FRAME:
                continue
            tag = tag_key(tf.child_frame_id, self.mode)
            if tag is None:
                continue
            stamp = tf.header.stamp.sec * 10**9 + tf.header.stamp.nanosec
            if stamp < self.after_ns or not 0 <= now_ns - stamp <= self.MAX_TAG_LAG_NS:
                continue
            p = tf.transform.translation
            if not (math.isfinite(p.x) and 0.20 < p.z < 2.8 and abs(p.x) < 0.85 * p.z):
                continue
            self.readings.setdefault(tag, {})[stamp] = (p.z, -p.x)

    def stopped_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            self.command()
            rclpy.spin_once(self, timeout_sec=0.04)

    def collect(self, seconds):
        """Stand still and return the median reading of each tag seen at least twice."""
        self.after_ns = None
        self.stopped_for(0.25)
        self.readings.clear()
        self.after_ns = self.get_clock().now().nanoseconds
        self.stopped_for(seconds)
        self.after_ns = None
        return [(tag, np.median(list(readings.values()), axis=0))
                for tag, readings in self.readings.items() if len(readings) >= 2]

    def look(self, seconds=0.55):
        for tag, observation in sorted(self.collect(seconds),
                                       key=lambda item: item[0] not in arena.BOUNDARY_TAGS):
            if self.filter.observe(tag, observation) and tag in arena.BOUNDARY_TAGS:
                self.last_anchor = time.monotonic()
                self.anchors.add(tag)
        self.trace.append(self.filter.x[:3].tolist())

    def pulse(self, distance=0.0, turn=0.0):
        pose = self.filter.x[:3]
        sigma = self.filter.position_sigma()
        heading_sigma = math.sqrt(max(0., self.filter.P[2, 2]))
        allowance = arena.ROBOT_RADIUS + 0.025 + 3 * sigma
        end = pose[:2]
        if distance:
            allowance += distance * min(1., 3 * heading_sigma)
            end = end + distance * np.array([math.cos(pose[2]), math.sin(pose[2])])
        if clearance(pose, end, self.cylinder) < allowance:
            raise RuntimeError("Available clearance is too small for the next pulse.")
        if distance:
            left = right = arena.FORWARD_CMD
            duration = distance / arena.FORWARD_SPEED
        else:
            rate = (math.pi / 2) / (arena.LEFT_90_S if turn > 0 else arena.RIGHT_90_S)
            command = arena.LEFT_CMD if turn > 0 else -arena.RIGHT_CMD
            left, right = -command, command
            duration = abs(turn) / rate

        start = time.monotonic()
        if self.drive_started is None:
            self.drive_started = start
        try:
            while time.monotonic() < start + duration:
                self.check()
                self.command(left, right)
                time.sleep(min(0.04, max(0., start + duration - time.monotonic())))
        finally:
            if rclpy.ok():
                self.command()
        if time.monotonic() - start > duration + 0.10:
            raise RuntimeError("Motor pulse timing overrun.")
        self.wheel_model_distance += distance
        self.filter.predict(distance, turn)
        self.look()

    def localized(self):
        return (len(self.anchors) >= 2 and time.monotonic() - self.last_anchor < 6.
                and self.filter.position_sigma() < 0.055 and self.filter.P[2, 2] < 0.18**2)

    def scan(self):
        for _ in range(12):
            self.pulse(turn=math.pi / 6)
            if self.localized():
                return
        raise RuntimeError("Could not localize using two boundary tags.")

    def go(self, target):
        for _ in range(100):
            if not self.localized():
                self.scan()
            pose = self.filter.x[:3]
            dx, dy = np.asarray(target) - pose[:2]
            distance = math.hypot(dx, dy)
            if distance <= 0.035:
                return
            error = wrap(math.atan2(dy, dx) - pose[2])
            if abs(error) > 0.075:
                self.pulse(turn=math.copysign(min(abs(error), math.pi / 6), error))
            else:
                self.pulse(distance=min(distance, arena.STEP_M))
        raise RuntimeError(f"Could not reach waypoint {target}.")

    def drive_to(self, target, method):
        route, self.nodes, self.edges = plan(self.filter.x[:2], target, self.cylinder, method)
        self.route.extend(route)
        for waypoint in route[1:]:
            self.go(waypoint)

    def face_goal(self):
        """Best-effort turn to face +x at the goal, re-measuring after each turn."""
        for _ in range(6):
            error = wrap(0.0 - self.filter.x[2])
            if abs(error) <= 0.05:
                break
            self.pulse(turn=math.copysign(min(abs(error), math.pi / 6), error))

    def start(self):
        for _ in range(10):
            self.look(0.65)
            if self.motor.get_subscription_count() and self.anchors:
                break
        if not self.motor.get_subscription_count():
            raise RuntimeError("No subscriber to /motor_commands. Check motor.log.")

    def run(self):
        self.start()
        if not self.localized():
            self.scan()
        self.drive_to(arena.GOAL, self.mode if self.mode in ROUTES else "safety")
        self.face_goal()

    def result(self, error):
        route = self.route
        return {
            "status": "stopped" if error else "completed",
            "error": str(error) if error else None,
            "elapsed_s": time.monotonic() - self.begin,
            "since_first_motion_s": time.monotonic() - self.drive_started if self.drive_started else 0.,
            "estimated_final_pose": self.filter.x[:3].tolist(),
            "estimated_goal_error_m": (math.dist(self.filter.x[:2], arena.GOAL)
                                       if self.mode != "survey" else None),
            "planned_distance_m": sum(math.dist(a, b) for a, b in zip(route, route[1:])),
            "planned_body_clearance_m": min((clearance(a, b, self.cylinder) - arena.ROBOT_RADIUS
                                            for a, b in zip(route, route[1:])), default=None),
            "commanded_distance_m": self.wheel_model_distance,
            "planned_path": route,
            "estimated_trace": self.trace,
            "cylinder_used": self.cylinder,
            "mapped_tags": {tag: self.filter.x[i:i+2].tolist() for tag, i in self.filter.indices.items()},
        }

    def save(self, error):
        save_json(self.folder / "result.json", self.result(error))
        plot(self.folder / "path.svg", self.mode, self.route, self.nodes, self.edges,
             self.cylinder, self.trace)


class SlamRobot(Robot):
    MAX_TAG_LAG_NS = 1_500_000_000

    def __init__(self, mode, cylinder, folder, children):
        super().__init__(mode, cylinder, folder, children)
        self.tag_stats = {}
        self.sightings = []
        self.rejected_looks = 0
        self.relocalizations = 0

    def look(self, seconds=0.55):
        items = self.collect(seconds)
        wall = [(t, o) for t, o in items if t in arena.BOUNDARY_TAGS]
        wall_used = sum(self.use_reading(t, o) for t, o in wall)
        if wall and not wall_used:
            self.rejected_looks += 1
            if self.rejected_looks >= 2 and self.relocalize_heading(wall):
                wall_used = sum(self.use_reading(t, o) for t, o in wall)
        if wall_used:
            self.rejected_looks = 0
        for t, o in items:
            if t not in arena.BOUNDARY_TAGS:
                self.use_reading(t, o)
        self.trace.append(self.filter.x[:3].tolist())

    def use_reading(self, tag, observation):
        pose = self.filter.x[:3].copy()
        camera_xy = pose[:2] + rotation(pose[2]) @ self.filter.offset
        world = camera_xy + rotation(pose[2] + arena.CAMERA_YAW) @ observation
        accepted = self.filter.observe(tag, observation)
        self.tag_stats.setdefault(str(tag), [0, 0])[0 if accepted else 1] += 1
        self.sightings.append({"tag": str(tag), "robot_pose": pose.tolist(),
                               "camera_reading": observation.tolist(),
                               "implied_tag_xy": world.tolist(), "accepted": accepted})
        if accepted and tag in arena.BOUNDARY_TAGS:
            self.last_anchor = time.monotonic()
            self.anchors.add(tag)
        return accepted

    def relocalize_heading(self, wall):
        headings = []
        for tag, z in wall:
            target = np.asarray(arena.BOUNDARY_TAGS[tag])
            heading = self.filter.x[2]
            for _ in range(3):
                camera = self.filter.x[:2] + rotation(heading) @ self.filter.offset
                bearing = math.atan2(*(target - camera)[::-1])
                heading = wrap(bearing - math.atan2(z[1], z[0]) - arena.CAMERA_YAW)
            camera = self.filter.x[:2] + rotation(heading) @ self.filter.offset
            if abs(np.linalg.norm(target - camera) - np.linalg.norm(z)) < 0.35:
                headings.append(heading)
        if not headings:
            return False
        mean = math.atan2(sum(map(math.sin, headings)), sum(map(math.cos, headings)))
        if max(abs(wrap(h - mean)) for h in headings) > 0.2:
            return False
        self.filter.x[2] = mean
        self.filter.P[2, 2] = max(self.filter.P[2, 2], 0.15**2)
        self.filter.P[0, 0] += 0.04**2
        self.filter.P[1, 1] += 0.04**2
        self.rejected_looks = 0
        self.relocalizations += 1
        return True

    def localized(self):
        return (len(self.anchors) >= 2 and self.filter.position_sigma() < 0.055
                and self.filter.P[2, 2] < 0.18**2)

    def face(self, heading):
        for _ in range(20):
            if not self.localized():
                self.scan()
            error = wrap(heading - self.filter.x[2])
            if abs(error) <= 0.075:
                return
            self.pulse(turn=math.copysign(min(abs(error), math.pi / 6), error))
        raise RuntimeError("Could not turn to the requested heading.")

    def sweep_cylinder(self):
        """Look left of, at, and right of the cylinder."""
        dx, dy = np.asarray(self.cylinder[:2]) - self.filter.x[:2]
        centre = math.atan2(dy, dx)
        for offset in (-0.35, 0.0, 0.35):
            self.face(centre + offset)
            self.look(2.0)

    def return_home(self):
        route, _, _ = plan(self.filter.x[:2], arena.START[:2], self.cylinder, "safety")
        for waypoint in route[1:]:
            self.go(waypoint)
        self.face(arena.START[2])
        self.look(0.65)

    def survey(self):
        self.start()
        if not self.localized():
            self.scan()
        for stop in arena.SURVEY_STOPS:
            self.drive_to(stop, "safety")
            self.sweep_cylinder()
        self.mapped = self.filter.cylinder()
        save_json(self.folder / "mapped_cylinder.json", self.mapped)
        plot(self.folder / "slam.svg", "SLAM map", [], [], [],
             (*self.mapped["center"], self.mapped["radius"]), self.trace, self.mapped["tags"])
        self.return_home()

    def run(self):
        if self.mode == "survey":
            self.survey()
        else:
            super().run()

    def result(self, error):
        result = super().result(error)
        result.update(tag_accepted_rejected=self.tag_stats, tag_sightings=self.sightings,
                      relocalizations=self.relocalizations)
        return result

    def save(self, error):
        super().save(error)
        sightings_plot(self.folder / "tag_sightings.svg", self.mode, self.sightings,
                       self.trace, self.cylinder)


def drive_robot(mode, cylinder, folder, children):
    folder.mkdir()
    rclpy.init(args=[])
    robot = None
    error = None
    try:
        robot = (SlamRobot if mode in ("survey", "mapped") else Robot)(mode, cylinder, folder, children)
        robot.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if robot is not None:
            try:
                if rclpy.ok():
                    robot.command()
                robot.save(error)
            finally:
                robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return robot.mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", *ROUTES, "slam"))
    mode = parser.parse_args().mode
    output = Path.home() / "hw4_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)

    if mode == "plan":
        for method in ROUTES:
            route, nodes, edges = plan(arena.START, arena.GOAL, arena.CYLINDER, method)
            plot(output / f"{method}.svg", method, route, nodes, edges, arena.CYLINDER)
        return

    mapped = None
    with hardware(output, mode == "slam") as children:
        for run in (("survey", "mapped") if mode == "slam" else (mode,)):
            cylinder = arena.CYLINDER
            if run == "mapped":
                cylinder = (*mapped["center"], mapped["radius"] + mapped["padding"])
            mapped = drive_robot(run, cylinder, output / run, children) or mapped


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
