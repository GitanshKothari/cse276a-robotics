"""Recover a robot pose from one AprilTag detection, with a noise model.

The detection message carries the four corners in pixels, so the pose is
solved here rather than taken from the tf the detector broadcasts. Solving
it locally buys two things the tf does not carry: the reprojection error,
and the second solution of the planar pose problem, which is what tells us
whether the tag's orientation can be trusted at all.
"""

import math

import cv2
import numpy as np

from .transforms import invert_transform, make_transform

# Corner order used by the AprilTag library, and therefore by the corners
# field of the detection message: counter-clockwise on the tag face with
# x to the right and y up. The vendored apriltag_ros pnp() pairs its object
# points with detection->p in exactly this order, so the two agree.
CORNER_ORDER = ((-1, -1), (1, -1), (1, 1), (-1, 1))

# IPPE_SQUARE is the only OpenCV solver that hands back both planar
# solutions for a square, but it requires its points in the fixed order
# top-left, top-right, bottom-right, bottom-left. Against CORNER_ORDER
# that is simply the reverse, applied to the object and image points
# together so the correspondence survives.
IPPE_SQUARE_ORDER = (3, 2, 1, 0)

# Two solutions whose rotations are closer than this are the same pose
# found twice, not a genuine front/back ambiguity.
DISTINCT_ROTATION_DEG = 10.0


class TagObservation:
    """One usable sighting of a mapped tag."""

    def __init__(self, tag_id, transform, distance, reprojection_error, ambiguity):
        self.tag_id = tag_id
        self.transform = transform                  # 4x4 camera <- tag
        self.distance = distance                    # metres, camera to tag
        self.reprojection_error = reprojection_error
        self.ambiguity = ambiguity                  # 1.0 == the two solutions tie


def object_points(size):
    half = 0.5 * size
    return np.array([[half * x, half * y, 0.0] for x, y in CORNER_ORDER], dtype=np.float64)


def _camera_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _reprojection_rms(object_pts, image_pts, camera_matrix, rvec, tvec):
    """RMS corner reprojection error in pixels.

    Computed here rather than taken from solvePnPGeneric because the error
    has to be comparable across solvers, and because the refinement step
    below moves each solution away from whatever error it was scored with.
    """
    projected, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, None)
    residual = projected.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt((residual ** 2).sum(axis=1).mean()))


def _refine(object_pts, image_pts, camera_matrix, rvec, tvec):
    """Polish one candidate with Levenberg-Marquardt."""
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_pts, image_pts, camera_matrix, None, rvec, tvec)
    except cv2.error:
        pass
    return rvec, tvec


def solve_tag_pose(corners, size, fx, fy, cx, cy):
    """Camera-from-tag transform for one detection.

    Returns ``(transform, reprojection_error, ambiguity)``, or None when the
    solver fails. ``ambiguity`` is the ratio of the best to the second-best
    reprojection error over genuinely different orientations: near 1 the two
    planar solutions fit the corners equally well and the recovered
    orientation is a coin flip.

    Three candidates are generated and all of them are refined: the two
    IPPE_SQUARE solutions, which are what expose the ambiguity, plus an
    SQPNP solution. IPPE on its own is not enough - on OpenCV 4.6 it
    returns rotations tens of degrees off even on noise-free corners, and
    close oblique views are exactly where both of its seeds land in a local
    minimum that the refinement cannot climb out of. SQPNP covers those,
    and scoring every refined candidate by the same reprojection measure
    picks the right one without trusting any single solver.
    """
    object_pts = object_points(size)
    image_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = _camera_matrix(fx, fy, cx, cy)

    # The image is rectified upstream, so there is no distortion left to undo.
    candidates = []

    permutation = list(IPPE_SQUARE_ORDER)
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            np.ascontiguousarray(object_pts[permutation]),
            np.ascontiguousarray(image_pts[permutation]),
            camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        count = 0
    for index in range(count):
        candidates.append(
            _refine(object_pts, image_pts, camera_matrix, rvecs[index], tvecs[index]))

    try:
        solved, rvec, tvec = cv2.solvePnP(
            object_pts, image_pts, camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP)
        if solved:
            candidates.append(
                _refine(object_pts, image_pts, camera_matrix, rvec, tvec))
    except cv2.error:
        pass

    # Drop anything that puts the tag behind the camera, then rank by fit.
    scored = [
        (_reprojection_rms(object_pts, image_pts, camera_matrix, rvec, tvec), rvec, tvec)
        for rvec, tvec in candidates
        if np.asarray(tvec).reshape(3)[2] > 1e-6
    ]
    if not scored:
        return None
    scored.sort(key=lambda candidate: candidate[0])

    error, rvec, tvec = scored[0]
    rotation, _ = cv2.Rodrigues(rvec)

    # The runner-up has to be a different orientation, or the ratio ends up
    # comparing the winner against a duplicate of itself and every tag
    # looks hopelessly ambiguous.
    ambiguity = 0.0
    for other_error, other_rvec, _ in scored[1:]:
        other_rotation, _ = cv2.Rodrigues(other_rvec)
        cosine = 0.5 * (float(np.trace(rotation.T @ other_rotation)) - 1.0)
        separation = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if separation < DISTINCT_ROTATION_DEG:
            continue
        if other_error > 1e-9:
            ambiguity = error / other_error
        break

    transform = make_transform(rotation, np.asarray(tvec).reshape(3))
    return transform, error, ambiguity


def robot_pose_from_tag(camera_from_tag, world_from_tag, base_from_camera):
    """World pose of base_link implied by seeing a tag of known pose."""
    # world <- tag  ==  world <- base  *  base <- camera  *  camera <- tag
    return world_from_tag @ invert_transform(base_from_camera @ camera_from_tag)


def tag_in_base(camera_from_tag, base_from_camera):
    """Position of the tag centre in base_link coordinates."""
    point = base_from_camera @ np.append(camera_from_tag[:3, 3], 1.0)
    return point[:3]


def range_bearing(position_in_base):
    """Horizontal range and bearing of a tag seen at ``position_in_base``."""
    x, y = float(position_in_base[0]), float(position_in_base[1])
    return math.hypot(x, y), math.atan2(y, x)


def range_bearing_noise(distance, focal, size, pixel_sigma,
                        floor_range, floor_bearing):
    """Covariance of one (range, bearing) observation.

    Bearing comes from where the tag sits in the image, which four averaged
    corners localise to well under a pixel and which does not get worse with
    distance. Range comes from how large the tag looks, and the apparent
    size falls off as 1/d, so the range error grows with the square of the
    distance. These two differ by more than an order of magnitude at normal
    operating ranges, which is the whole reason the tag is fused as range
    and bearing instead of as a full pose.
    """
    range_sigma = math.hypot((distance * distance / (focal * size)) * pixel_sigma,
                             floor_range)
    bearing_sigma = math.hypot(pixel_sigma / focal, floor_bearing)
    return np.diag([range_sigma ** 2, bearing_sigma ** 2])


def yaw_noise(distance, focal, size, pixel_sigma, scale,
              ambiguity, ambiguity_threshold, floor_yaw):
    """Variance of the heading implied by a tag's recovered orientation.

    Which way a small square faces is read off its perspective
    foreshortening, and that is second order in the viewing angle near
    head-on. Measured against a synthetic ground truth the error is about
    ten times what the first-order pixel-to-angle scaling predicts, hence
    ``scale``. When the two planar solutions fit the corners nearly as well
    as each other the orientation is close to a coin flip, so it is widened
    until the filter effectively ignores it.
    """
    sigma = math.hypot(scale * (distance / (focal * size)) * pixel_sigma, floor_yaw)
    if ambiguity > ambiguity_threshold:
        sigma *= 10.0
    return sigma ** 2
