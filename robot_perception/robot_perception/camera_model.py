"""Pinhole geometry: a box of known real size becomes a 3D point.

This is separate from the detector because it is the only part that needs
the camera calibration. It is separate from the node because it is pure
arithmetic, so it can be tested without ROS and without a camera.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics of a rectified camera, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_projection_matrix(cls, matrix: Sequence[float]) -> 'PinholeCamera':
        """Read the intrinsics out of ``camera_info.p``.

        P, not K. The camera node rectifies every frame and republishes
        camera_info with zero distortion, so P holds the intrinsics that
        match the pixels which actually arrive. K describes the raw lens.
        """
        return cls(fx=float(matrix[0]), fy=float(matrix[5]),
                   cx=float(matrix[2]), cy=float(matrix[6]))

    @property
    def is_valid(self) -> bool:
        return self.fx > 0.0 and self.fy > 0.0

    def depth_from_height(self, pixel_height: float, real_height: float) -> float:
        """Distance along the optical axis, by triangle similarity.

        Apparent height falls off as 1/depth, so depth = f * H / h. The
        range error therefore grows with the square of the depth. That is
        why a can is worth using at short range only.

        Raises ValueError if the box has no height.
        """
        if pixel_height <= 0.0:
            raise ValueError('a box with no height has no depth')
        return self.fy * real_height / float(pixel_height)

    def back_project(self, pixel: tuple[float, float],
                     depth: float) -> tuple[float, float, float]:
        """A pixel at a known depth, in the camera optical frame.

        OpenCV axes: x to the right, y down, z along the optical axis. This
        is the frame ``camera_extrinsics`` produces, so the localizer reuses
        ``base_from_camera`` for a can exactly as it does for a tag.
        """
        u, v = pixel
        return ((u - self.cx) * depth / self.fx,
                (v - self.cy) * depth / self.fy,
                depth)
