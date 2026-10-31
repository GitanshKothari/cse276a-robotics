"""The seam between how a can is found and how it is used.

The node depends on this protocol, never on a particular detector. A
learned detector can take the colour one's place without a change to the
node. That is why the protocol stays this small: one method, pixels in,
boxes out. It knows nothing about ROS and nothing about the camera.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One candidate can, in the pixels of the frame that was searched."""

    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + 0.5 * self.width, self.y + 0.5 * self.height

    def scaled(self, factor: float) -> 'Detection':
        """The same box, in a frame ``factor`` times larger.

        The node searches a reduced frame for speed, then scales the boxes
        back before it does any geometry. Confidence does not change.
        """
        return Detection(
            x=round(self.x * factor),
            y=round(self.y * factor),
            width=round(self.width * factor),
            height=round(self.height * factor),
            confidence=self.confidence,
        )


class Detector(Protocol):
    """Anything that finds cans in a frame."""

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Candidate cans in one BGR frame, best confidence first."""
