"""Find coke cans by the color of the label.

This detector needs no training data. It runs at frame rate on the board.
It has few parameters, so the report can give its accuracy.

Three facts about a coke can give the shape of the pipeline.

Red is at the two ends of the hue circle. OpenCV hue goes from 0 to 179,
and the label is at both ends. The mask therefore needs two ranges.

The white script cuts the red band into parts. A wide close kernel joins
the parts again. Without it, one can gives several thin blobs.

The ends of a can are bare aluminum. They are not red, so the blob shows
the label only. Every size in this module is a size of the label.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass(frozen=True)
class HsvCanConfig:
    """Thresholds, in the pixels of the frame that the detector searches."""

    # Red is at the two ends of the hue circle. The mask keeps hue 0 to
    # hue_low_max, and also hue_high_min to 179.
    hue_low_max: int = 10
    hue_high_min: int = 170
    saturation_min: int = 120
    value_min: int = 45

    blur_kernel: int = 5
    open_kernel: int = 3        # removes small specks
    close_kernel: int = 9       # joins the parts of the red band

    min_height: int = 12
    min_area: int = 120

    # Label height divided by can diameter. The label goes around the
    # cylinder, so its width in the image is the diameter from all sides.
    # Measured on real frames, not derived from the can's sizes. A
    # cylinder's edges curve away, go dim and drop out of the red mask, so
    # a blob reads narrower, and so taller, than the can really is.
    # Geometry wants 1.29 for a 90 x 70 mm label. A real can measures 1.75.
    target_aspect: float = 1.75
    aspect_tolerance: float = 0.30
    # 0.65 separates a coke can (measured 0.80) from a part-red can (0.54).
    fill_min: float = 0.65

    # The edge of the frame cuts a blob short. Its height, and thus its
    # depth, is then wrong. The detector removes such a blob.
    edge_margin: int = 2

    max_detections: int = 8


def _odd(value):
    """Make a kernel size odd and not less than 1."""
    value = int(value)
    if value < 1:
        return 1
    if value % 2 == 0:
        return value + 1
    return value


def _morph(mask, operation, kernel):
    """Do one morphology step. A kernel less than 3 does nothing."""
    if kernel < 3:
        return mask
    size = _odd(kernel)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, operation, element)


class HsvCanDetector:
    """Detector that thresholds the red label. It satisfies ``Detector``."""

    def __init__(self, config=None):
        if config is None:
            config = HsvCanConfig()
        self.config = config

    def detect(self, frame):
        """Candidate cans in one BGR frame, best confidence first."""
        mask = self.red_mask(frame)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = []
        for contour in contours:
            candidate = self._candidate(contour, frame.shape[0])
            if candidate is not None:
                found.append(candidate)

        found.sort(key=lambda item: item.confidence, reverse=True)
        return found[:self.config.max_detections]

    def red_mask(self, frame):
        """Binary mask of the red label. Public, because tuning needs it."""
        config = self.config
        if config.blur_kernel >= 3:
            size = _odd(config.blur_kernel)
            frame = cv2.GaussianBlur(frame, (size, size), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = config.saturation_min
        value = config.value_min

        # One range at each end of the hue circle.
        low = cv2.inRange(
            hsv,
            np.array([0, saturation, value], np.uint8),
            np.array([config.hue_low_max, 255, 255], np.uint8))
        high = cv2.inRange(
            hsv,
            np.array([config.hue_high_min, saturation, value], np.uint8),
            np.array([179, 255, 255], np.uint8))

        mask = cv2.bitwise_or(low, high)
        mask = _morph(mask, cv2.MORPH_OPEN, config.open_kernel)
        mask = _morph(mask, cv2.MORPH_CLOSE, config.close_kernel)
        return mask

    def _candidate(self, contour, frame_height):
        """One detection, if the contour passes every gate. Else None."""
        config = self.config

        x, y, width, height = cv2.boundingRect(contour)
        if width < 1 or height < config.min_height:
            return None

        area = cv2.contourArea(contour)
        if area < config.min_area:
            return None

        # The edge of the frame cuts this blob, so its height is too small.
        margin = config.edge_margin
        if y <= margin or y + height >= frame_height - margin:
            return None

        aspect = height / float(width)
        error = abs(aspect - config.target_aspect) / config.target_aspect
        if error > config.aspect_tolerance:
            return None

        fill = area / float(width * height)
        if fill < config.fill_min:
            return None

        # A good blob has the correct shape, and it is also full of red.
        shape = 1.0 - error / config.aspect_tolerance
        confidence = max(0.0, shape) * fill
        return Detection(x=x, y=y, width=width, height=height,
                         confidence=confidence)
