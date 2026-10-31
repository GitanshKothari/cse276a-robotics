"""Detector gates, checked on synthetic frames.

A real can is not a flat red rectangle, so these do not prove the
thresholds suit your lighting. They prove the shape gates behave: the
right shape passes, the wrong shape does not, and a clipped blob is
dropped rather than reported at a wrong distance.
"""

import numpy as np
import pytest

from robot_perception.hsv_can_detector import HsvCanConfig, HsvCanDetector

FRAME_HEIGHT = 480
FRAME_WIDTH = 640
RED = (0, 0, 255)       # BGR


def frame_with(*boxes):
    """A black frame with a filled red rectangle per (x, y, width, height)."""
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    for x, y, width, height in boxes:
        canvas[y:y + height, x:x + width] = RED
    return canvas


def test_finds_a_label_shaped_blob():
    # 40 x 70 is an aspect of 1.75, which is what a real can measures.
    detector = HsvCanDetector()
    found = detector.detect(frame_with((100, 100, 40, 70)))

    assert len(found) == 1
    assert found[0].height / found[0].width == pytest.approx(1.75, abs=0.2)
    assert found[0].confidence > 0.5


def test_rejects_a_blob_of_the_wrong_shape():
    # Wider than tall, so the aspect error is past the tolerance.
    detector = HsvCanDetector()
    assert detector.detect(frame_with((100, 100, 80, 40))) == []


def test_rejects_a_blob_against_the_top_edge():
    # Cut short by the frame, so its height, and so its depth, is wrong.
    detector = HsvCanDetector()
    assert detector.detect(frame_with((100, 0, 40, 70))) == []


def test_rejects_a_blob_against_the_bottom_edge():
    detector = HsvCanDetector()
    box = (100, FRAME_HEIGHT - 70, 40, 70)
    assert detector.detect(frame_with(box)) == []


def test_ignores_colours_that_are_not_red():
    detector = HsvCanDetector()
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    canvas[100:170, 100:140] = (255, 0, 0)      # blue
    assert detector.detect(canvas) == []


def test_finds_two_cans_and_orders_them_by_confidence():
    detector = HsvCanDetector()
    found = detector.detect(frame_with((60, 100, 40, 70), (300, 150, 40, 70)))

    assert len(found) == 2
    assert found[0].confidence >= found[1].confidence


def test_honours_the_detection_limit():
    detector = HsvCanDetector(HsvCanConfig(max_detections=1))
    found = detector.detect(frame_with((60, 100, 40, 70), (300, 150, 40, 70)))
    assert len(found) == 1
