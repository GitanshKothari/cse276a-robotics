"""Pinhole arithmetic, checked against figures worked out by hand."""

import pytest

from robot_perception.camera_model import PinholeCamera

# The rover's calibrated values.
CAMERA = PinholeCamera(fx=637.5, fy=637.3, cx=667.2, cy=318.8)
LABEL = 0.090


def test_reads_projection_matrix():
    # camera_info.p is a 3x4 row-major matrix: fx, 0, cx, 0, 0, fy, cy, 0...
    matrix = [637.5, 0.0, 667.2, 0.0,
              0.0, 637.3, 318.8, 0.0,
              0.0, 0.0, 1.0, 0.0]
    assert PinholeCamera.from_projection_matrix(matrix) == CAMERA


def test_depth_falls_off_as_one_over_height():
    # A label 1 m away stands fy * H = 57.36 px tall.
    assert CAMERA.depth_from_height(57.357, LABEL) == pytest.approx(1.0, abs=1e-3)
    # Half the height means twice the distance.
    assert CAMERA.depth_from_height(28.678, LABEL) == pytest.approx(2.0, abs=1e-3)


def test_depth_rejects_an_empty_box():
    with pytest.raises(ValueError):
        CAMERA.depth_from_height(0.0, LABEL)


def test_centre_pixel_lies_on_the_optical_axis():
    point = CAMERA.back_project((CAMERA.cx, CAMERA.cy), 1.5)
    assert point == pytest.approx((0.0, 0.0, 1.5))


def test_one_focal_length_off_centre_is_one_depth_sideways():
    # A pixel fx to the right of the centre subtends 45 degrees, so its
    # back-projection has x equal to its depth.
    point = CAMERA.back_project((CAMERA.cx + CAMERA.fx, CAMERA.cy), 2.0)
    assert point == pytest.approx((2.0, 0.0, 2.0))


def test_validity_needs_both_focal_lengths():
    assert CAMERA.is_valid
    assert not PinholeCamera(0.0, 637.3, 667.2, 318.8).is_valid
    assert not PinholeCamera(637.5, 0.0, 667.2, 318.8).is_valid
