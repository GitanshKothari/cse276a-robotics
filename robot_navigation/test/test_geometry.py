"""Synthetic checks on the pose pipeline.

A known rover pose and a known tag are projected to pixels with a pinhole
model, then pushed back through the same code the localizer uses. With
noise-free corners the recovered pose has to come back exact; with pixel
noise the spread has to match the noise model in tag_pose, because the EKF
weights every sighting by that model.
"""

import math

import numpy as np
import pytest

from robot_navigation.tag_pose import (
    object_points, range_bearing, range_bearing_noise, robot_pose_from_tag,
    solve_tag_pose, tag_in_base,
)
from robot_navigation.transforms import (
    camera_extrinsics, invert_transform, make_transform, planar_pose, rot_z,
    tag_extrinsics, wrap_angle,
)

FX = FY = 900.0
CX, CY = 640.0, 360.0
TAG_SIZE = 0.165

BASE_FROM_CAMERA = camera_extrinsics(0.10, 0.0, 0.18, 0.0, 0.0, 0.0)

# Rover pose, tag position and tag facing, sampled along the HW2 route.
GEOMETRIES = [
    ((0.0, 0.0, 0.0), (2.20, 0.00), 180.0),
    ((1.0, 0.0, 0.0), (2.20, 0.00), 180.0),
    ((0.5, 0.8, math.radians(-30)), (2.20, 0.00), 180.0),
    ((0.0, 0.0, math.radians(45)), (2.20, 1.20), 180.0),
    ((1.0, 1.0, math.radians(90)), (1.20, 2.80), 270.0),
    ((1.0, 2.0, math.radians(90)), (1.20, 2.80), 270.0),
    ((0.5, 1.0, math.radians(210)), (-0.80, 0.40), 0.0),
    ((0.0, 0.0, math.radians(180)), (-0.80, 0.40), 0.0),
]


def world_from_base(x, y, theta):
    return make_transform(rot_z(theta), (x, y, 0.0))


def setup_case(rover, tag_xy, normal_yaw_deg, tag_z=0.25):
    """Ground-truth camera-from-tag and world-from-tag for one geometry."""
    world_from_tag = tag_extrinsics(
        tag_xy[0], tag_xy[1], tag_z, math.radians(normal_yaw_deg))
    camera_from_tag = (
        invert_transform(BASE_FROM_CAMERA)
        @ invert_transform(world_from_base(*rover))
        @ world_from_tag
    )
    return camera_from_tag, world_from_tag


def project(camera_from_tag, size, pixel_noise=0.0, rng=None):
    """Pinhole projection of the four tag corners, in detection order."""
    pixels = []
    for point in object_points(size):
        camera_point = camera_from_tag @ np.append(point, 1.0)
        x, y, z = camera_point[:3]
        pixels.append((FX * x / z + CX, FY * y / z + CY))
    pixels = np.asarray(pixels)
    if pixel_noise:
        pixels = pixels + rng.normal(0.0, pixel_noise, pixels.shape)
    return pixels


@pytest.mark.parametrize('rover,tag_xy,normal_yaw', GEOMETRIES)
def test_noise_free_pose_is_exact(rover, tag_xy, normal_yaw):
    """Noise-free corners must recover the rover pose to numerical precision.

    This is the regression guard for the solver: IPPE alone used to come
    back tens of degrees off here, which both corrupted the single-tag
    heading and pushed honest detections past the reprojection gate.
    """
    camera_from_tag, world_from_tag = setup_case(rover, tag_xy, normal_yaw)
    assert camera_from_tag[2, 3] > 0.0, 'geometry puts the tag behind the camera'

    solved = solve_tag_pose(project(camera_from_tag, TAG_SIZE),
                            TAG_SIZE, FX, FY, CX, CY)
    assert solved is not None
    transform, error, _ = solved

    assert error < 1e-6

    recovered = planar_pose(
        robot_pose_from_tag(transform, world_from_tag, BASE_FROM_CAMERA))
    assert math.hypot(recovered[0] - rover[0], recovered[1] - rover[1]) < 1e-6
    assert abs(wrap_angle(recovered[2] - rover[2])) < 1e-6


@pytest.mark.parametrize('rover,tag_xy,normal_yaw', GEOMETRIES)
def test_range_and_bearing_match_geometry(rover, tag_xy, normal_yaw):
    """The range/bearing the EKF consumes must match the true geometry."""
    camera_from_tag, _ = setup_case(rover, tag_xy, normal_yaw)
    solved = solve_tag_pose(project(camera_from_tag, TAG_SIZE),
                            TAG_SIZE, FX, FY, CX, CY)
    assert solved is not None

    measured = range_bearing(tag_in_base(solved[0], BASE_FROM_CAMERA))
    expected = range_bearing(tag_in_base(camera_from_tag, BASE_FROM_CAMERA))
    assert measured[0] == pytest.approx(expected[0], abs=1e-6)
    assert abs(wrap_angle(measured[1] - expected[1])) < 1e-6


def test_noise_model_brackets_the_measured_spread():
    """The modelled range/bearing sigmas must track what noise actually does.

    The EKF trusts range_bearing_noise to weight every sighting, so the
    model is checked against a Monte Carlo rather than assumed. Range error
    grows with the square of the distance and bearing error does not grow
    at all, which is the whole reason a tag is fused as range and bearing
    instead of as a full pose.
    """
    rng = np.random.default_rng(7)
    focal = 0.5 * (FX + FY)
    bearing_spreads = []

    for rover, tag_xy, normal_yaw in GEOMETRIES:
        camera_from_tag, _ = setup_case(rover, tag_xy, normal_yaw)
        truth = range_bearing(tag_in_base(camera_from_tag, BASE_FROM_CAMERA))
        distance = float(np.linalg.norm(camera_from_tag[:3, 3]))

        range_errors, bearing_errors = [], []
        for _ in range(200):
            solved = solve_tag_pose(
                project(camera_from_tag, TAG_SIZE, 1.0, rng),
                TAG_SIZE, FX, FY, CX, CY)
            if solved is None:
                continue
            measured = range_bearing(tag_in_base(solved[0], BASE_FROM_CAMERA))
            range_errors.append(measured[0] - truth[0])
            bearing_errors.append(wrap_angle(measured[1] - truth[1]))

        covariance = range_bearing_noise(
            distance, focal, TAG_SIZE, 1.0, 0.005, math.radians(0.3))
        modelled_range = math.sqrt(covariance[0, 0])
        modelled_bearing = math.sqrt(covariance[1, 1])

        measured_range = float(np.std(range_errors))
        measured_bearing = float(np.std(bearing_errors))

        # The model must be the right order of magnitude and must not be
        # optimistic, since an optimistic sigma makes the filter overtrust.
        assert measured_range < modelled_range * 1.5
        assert measured_range > modelled_range * 0.2
        assert measured_bearing < modelled_bearing

        bearing_spreads.append(measured_bearing)

    # Bearing accuracy must not decay with distance the way range does.
    assert max(bearing_spreads) < 2.0 * min(bearing_spreads)


def test_single_tag_yaw_is_far_worse_than_bearing():
    """Justifies down-weighting the tag's own orientation.

    A tag's facing is read off its perspective foreshortening, which is
    second order in the viewing angle, so it is far noisier than the
    bearing. The localizer only falls back on it when a single tag is all
    there is, and this is the measurement behind that choice.
    """
    rng = np.random.default_rng(11)
    rover, tag_xy, normal_yaw = (0.0, 0.0, 0.0), (2.20, 0.00), 180.0
    camera_from_tag, world_from_tag = setup_case(rover, tag_xy, normal_yaw)

    yaw_errors, bearing_errors = [], []
    truth = range_bearing(tag_in_base(camera_from_tag, BASE_FROM_CAMERA))
    for _ in range(200):
        solved = solve_tag_pose(project(camera_from_tag, TAG_SIZE, 1.0, rng),
                                TAG_SIZE, FX, FY, CX, CY)
        if solved is None:
            continue
        recovered = planar_pose(
            robot_pose_from_tag(solved[0], world_from_tag, BASE_FROM_CAMERA))
        yaw_errors.append(wrap_angle(recovered[2] - rover[2]))
        measured = range_bearing(tag_in_base(solved[0], BASE_FROM_CAMERA))
        bearing_errors.append(wrap_angle(measured[1] - truth[1]))

    assert np.std(yaw_errors) > 50.0 * np.std(bearing_errors)
