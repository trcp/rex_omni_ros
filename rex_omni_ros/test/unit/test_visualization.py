"""Tests for the debug rendering of inference results."""

from __future__ import annotations

import numpy as np
from PIL import Image

from rex_omni_ros.core import visualization
from rex_omni_ros.core.engine import InferenceResult
from rex_omni_ros.core.types import (
    Annotation,
    Box,
    Keypoint,
    KeypointInstance,
    Point,
    Polygon,
)


def full_result() -> InferenceResult:
    return InferenceResult(
        annotations=[
            Annotation("cat", Box(10, 10, 80, 60), confidence=0.9),
            Annotation("dot", Point(150, 50), confidence=0.5),
            Annotation(
                "poly",
                Polygon([Point(100, 10), Point(180, 10), Point(180, 40)]),
                confidence=0.7,
            ),
        ],
        keypoint_instances=[
            KeypointInstance(
                category="person",
                box=Box(20, 20, 90, 90),
                keypoints=[
                    Keypoint("nose", Point(50, 50)),
                    Keypoint("left ear", None),
                ],
                confidence=1.0,
            )
        ],
    )


def test_render_draws_every_shape_kind_on_a_copy() -> None:
    image = Image.new("RGB", (200, 100))

    rendered = visualization.render_result(image, full_result())

    assert rendered is not image
    assert rendered.size == image.size
    assert np.asarray(rendered).any(), "nothing was drawn"
    assert not np.asarray(image).any(), "the input image was modified"


def test_render_empty_result_is_a_plain_copy() -> None:
    image = Image.new("RGB", (64, 48))

    rendered = visualization.render_result(image, InferenceResult())

    assert np.array_equal(np.asarray(rendered), np.asarray(image))
