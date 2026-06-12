"""End-to-end smoke tests with the real model on a CUDA GPU.

Slow: loads the full Rex-Omni model (downloads ~7 GB on first run).
Run via ``pixi run test-gpu``.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from rex_omni_ros.core.engine import EngineConfig, InferenceRequest, RexOmniEngine
from rex_omni_ros.core.tasks import TaskType
from rex_omni_ros.core.types import Box, Point

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

RECT = Box(150.0, 120.0, 420.0, 360.0)


@pytest.fixture(scope="module")
def engine():
    engine = RexOmniEngine(EngineConfig())
    engine.start()
    return engine


@pytest.fixture()
def test_image():
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((RECT.x0, RECT.y0, RECT.x1, RECT.y1), fill="red")
    return image


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    intersection = ix * iy
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0)
    return intersection / (union - intersection) if union else 0.0


def test_detection(engine, test_image):
    result = engine.infer(
        InferenceRequest(
            image=test_image,
            task=TaskType.DETECTION,
            categories=["red rectangle"],
        )
    )
    assert result.annotations, f"no detections; raw output: {result.raw_output!r}"
    boxes = [a.shape for a in result.annotations if isinstance(a.shape, Box)]
    assert boxes, "expected box-shaped detections"
    best = max(iou(box, RECT) for box in boxes)
    assert best > 0.5, f"poor localization (IoU {best:.2f}); raw: {result.raw_output!r}"
    assert all(0.0 <= a.confidence <= 1.0 for a in result.annotations)


def test_pointing(engine, test_image):
    result = engine.infer(
        InferenceRequest(
            image=test_image,
            task=TaskType.POINTING,
            categories=["red rectangle"],
        )
    )
    points = [a.shape for a in result.annotations if isinstance(a.shape, Point)]
    assert points, f"no points; raw output: {result.raw_output!r}"
    assert any(
        RECT.x0 <= p.x <= RECT.x1 and RECT.y0 <= p.y <= RECT.y1 for p in points
    ), f"point outside target; raw: {result.raw_output!r}"


def test_sleep_frees_vram_and_inference_survives_wake(engine, test_image):
    from rex_omni_ros.core.engine import GIB

    free_awake, _ = torch.cuda.mem_get_info()
    engine.sleep()
    free_asleep, _ = torch.cuda.mem_get_info()
    # The weights alone are several GiB; require a clearly visible drop.
    freed = free_asleep - free_awake
    assert freed > 2 * GIB, f"sleep freed only {freed / GIB:.2f} GiB"

    engine.wake_up()
    result = engine.infer(
        InferenceRequest(
            image=test_image,
            task=TaskType.DETECTION,
            categories=["red rectangle"],
        )
    )
    boxes = [a.shape for a in result.annotations if isinstance(a.shape, Box)]
    assert boxes, f"no detections after wake_up; raw: {result.raw_output!r}"
    best = max(iou(box, RECT) for box in boxes)
    assert best > 0.5, f"degraded localization after wake_up (IoU {best:.2f})"
