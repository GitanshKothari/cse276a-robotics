"""Check that the landmark map actually covers the planned route.

Where to put the tags is the one design choice in HW2 that cannot be tuned
once the rover is driving, so it is worth settling offline. This sweeps
every pose the controller drives through - the straight legs plus the
in-place turns at each waypoint - and counts the tags that are genuinely
detectable from each one: in front of the camera, with all four corners
inside the rectified image, within range, and not viewed so obliquely that
the corner fit falls apart.

    ros2 run robot_navigation check_coverage src/.../config/landmarks.yaml

A pose that sees nothing is a pose where the rover is navigating on dead
reckoning alone, and a run of them is where the error at the next waypoint
comes from.
"""

import argparse
import math
import os

import cv2
import numpy as np
import yaml

from .landmark_map import LandmarkMap
from .tag_pose import object_points
from .transforms import (
    camera_extrinsics, invert_transform, make_transform, rot_z, wrap_angle,
)

# Fall back on the OV5647 calibration shipped with robot_vision_camera.
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
DEFAULT_CAMERA_MATRIX = np.array([
    [490.980065, 1.448770, 639.427696],
    [0.0, 492.132778, 357.232355],
    [0.0, 0.0, 1.0],
])
DEFAULT_DISTORTION = np.array([0.487826, -1.851282, 1.209359, -0.068021, 0.0])


def rectified_intrinsics(path, width, height):
    """Intrinsics of the *rectified* image, which is what the detector sees.

    The camera node undistorts with getOptimalNewCameraMatrix and then
    advertises that matrix, so the raw calibration is the wrong thing to
    reason about field of view with.
    """
    camera_matrix, distortion = DEFAULT_CAMERA_MATRIX, DEFAULT_DISTORTION
    if path and os.path.exists(path):
        # OpenCV's own yaml flavour, which pyyaml cannot read.
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        try:
            matrix = storage.getNode('camera_matrix').mat()
            coefficients = storage.getNode('distortion_coefficients').mat()
            if matrix is not None:
                camera_matrix = matrix
            if coefficients is not None:
                distortion = coefficients.reshape(-1)
        finally:
            storage.release()

    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, (width, height), 0.0, (width, height), 0)
    return new_matrix


def tag_is_visible(rover, world_from_tag, size, base_from_camera, intrinsics,
                   width, height, max_range, max_oblique):
    """Would the detector get a usable look at this tag from this pose?"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    world_from_base = make_transform(rot_z(rover[2]), (rover[0], rover[1], 0.0))
    camera_from_tag = (
        invert_transform(base_from_camera)
        @ invert_transform(world_from_base)
        @ world_from_tag
    )

    if camera_from_tag[2, 3] <= 0.05:
        return False
    if float(np.linalg.norm(camera_from_tag[:3, 3])) > max_range:
        return False

    # How far off the tag's own normal the camera sits. Past roughly 65 deg
    # the face is foreshortened so hard that the corner fit degrades.
    camera_in_world = (world_from_base @ base_from_camera)[:3, 3]
    to_camera = camera_in_world - world_from_tag[:3, 3]
    norm = np.linalg.norm(to_camera)
    if norm < 1e-6:
        return False
    cosine = float((to_camera / norm) @ world_from_tag[:3, 2])
    if math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) > max_oblique:
        return False

    # Every corner has to land on the sensor, which is what actually
    # enforces the vertical field of view at close range.
    for point in object_points(size):
        camera_point = camera_from_tag @ np.append(point, 1.0)
        if camera_point[2] <= 0.05:
            return False
        u = fx * camera_point[0] / camera_point[2] + cx
        v = fy * camera_point[1] / camera_point[2] + cy
        if not (0.0 <= u < width and 0.0 <= v < height):
            return False
    return True


def route_poses(waypoints, turn_steps=40, drive_step=0.05):
    """Every pose the three-phase controller passes through."""
    poses = []
    current = waypoints[0]
    for target in waypoints[1:]:
        bearing = math.atan2(target[1] - current[1], target[0] - current[0])

        # Phase 1: turn on the spot onto the travel bearing.
        turn = wrap_angle(bearing - current[2])
        for fraction in np.linspace(0.0, 1.0, turn_steps):
            poses.append((current[0], current[1], current[2] + fraction * turn))

        # Phase 2: drive the leg holding that bearing.
        length = math.hypot(target[1] - current[1], target[0] - current[0])
        for fraction in np.linspace(0.0, 1.0, max(2, int(length / drive_step))):
            poses.append((
                current[0] + fraction * (target[0] - current[0]),
                current[1] + fraction * (target[1] - current[1]),
                bearing,
            ))

        # Phase 3: turn on the spot onto the heading the waypoint asks for.
        settle = wrap_angle(target[2] - bearing)
        for fraction in np.linspace(0.0, 1.0, turn_steps):
            poses.append((target[0], target[1], bearing + fraction * settle))

        current = target
    return poses


def blind_runs(counts):
    """Index ranges over which nothing at all is visible."""
    runs, start = [], None
    for index, count in enumerate(counts):
        if count == 0 and start is None:
            start = index
        elif count != 0 and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(counts) - 1))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_path', help='landmark yaml to check')
    parser.add_argument('--camera-config', default='',
                        help='robot_vision_camera camera_parameter.yaml')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--camera-x', type=float, default=0.10)
    parser.add_argument('--camera-y', type=float, default=0.0)
    parser.add_argument('--camera-z', type=float, default=0.18)
    parser.add_argument('--camera-yaw-deg', type=float, default=0.0)
    parser.add_argument('--camera-pitch-deg', type=float, default=0.0)
    parser.add_argument('--max-range', type=float, default=4.0)
    parser.add_argument('--max-oblique-deg', type=float, default=65.0)
    parser.add_argument('--waypoints', default='0,0,0 1,0,0 1,2,3.14159 0,0,0',
                        help='space separated x,y,theta triples')
    args = parser.parse_args(argv)

    waypoints = []
    for chunk in args.waypoints.split():
        parts = chunk.split(',')
        if len(parts) == 3:
            waypoints.append(tuple(float(p) for p in parts))
    if len(waypoints) < 2:
        raise SystemExit('need at least two waypoints')

    intrinsics = rectified_intrinsics(args.camera_config, args.width, args.height)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    horizontal = 2.0 * math.degrees(math.atan2(args.width / 2.0, fx))
    vertical = 2.0 * math.degrees(math.atan2(args.height / 2.0, fy))

    landmark_map = LandmarkMap.from_file(args.map_path)
    base_from_camera = camera_extrinsics(
        args.camera_x, args.camera_y, args.camera_z,
        math.radians(args.camera_yaw_deg), math.radians(args.camera_pitch_deg), 0.0)

    print(f'rectified fx={fx:.1f} fy={fy:.1f}, '
          f'field of view {horizontal:.1f} x {vertical:.1f} deg')
    print(f'{len(landmark_map)} landmarks: {landmark_map.ids()}')
    print(f'limits: {args.max_range:.1f} m range, '
          f'{args.max_oblique_deg:.0f} deg obliqueness\n')

    def visible_ids(pose):
        return [
            tag_id for tag_id in landmark_map.ids()
            if tag_is_visible(pose, landmark_map.get(tag_id).pose,
                              landmark_map.get(tag_id).size, base_from_camera,
                              intrinsics, args.width, args.height,
                              args.max_range, args.max_oblique_deg)
        ]

    poses = route_poses(waypoints)
    counts = np.array([len(visible_ids(pose)) for pose in poses])

    print(f'{len(poses)} poses swept along the route')
    for label, mask in (('no tag in view  ', counts == 0),
                        ('exactly one tag ', counts == 1),
                        ('two or more tags', counts >= 2)):
        print(f'  {label}: {mask.sum():4d}  ({100.0 * mask.mean():5.1f}%)')
    print(f'  mean tags in view: {counts.mean():.2f}')

    print('\nat each commanded waypoint:')
    for index, pose in enumerate(waypoints):
        ids = visible_ids(pose)
        print(f'  W{index} ({pose[0]:5.2f}, {pose[1]:5.2f}, '
              f'{math.degrees(pose[2]):7.1f} deg): {len(ids)} tags {ids}')

    runs = blind_runs(counts)
    if runs:
        print('\nblind stretches - the rover dead reckons through these:')
        for start, end in runs:
            first, last = poses[start], poses[end]
            print(f'  {end - start + 1} poses: '
                  f'({first[0]:.2f}, {first[1]:.2f}, {math.degrees(first[2]):6.1f} deg) '
                  f'-> ({last[0]:.2f}, {last[1]:.2f}, {math.degrees(last[2]):6.1f} deg)')
    else:
        print('\nno blind poses: every pose on the route sees at least one tag')

    return 0 if not runs else 1


if __name__ == '__main__':
    raise SystemExit(main())
