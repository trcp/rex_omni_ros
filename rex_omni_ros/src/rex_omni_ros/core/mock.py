"""Deterministic fake engine for tests and GPU-less dry runs.

Generates canned Rex-Omni token output for each task and runs it through the
real parser, so the full parse/convert path is exercised end to end.
"""

from __future__ import annotations

import json
import time

from rex_omni_ros.core import parser, tasks
from rex_omni_ros.core.engine import InferenceRequest, InferenceResult

_BOX_TOKENS = "<100><100><500><500>"
_POINT_TOKENS = "<250><250>"
_POLYGON_TOKENS = "<100><100><500><100><500><200><100><200>"


def _ref_block(category: str, coords: str) -> str:
    return (
        f"<|object_ref_start|>{category}<|object_ref_end|>"
        f"<|box_start|>{coords}<|box_end|>"
    )


class MockEngine:
    """Engine double; satisfies :class:`rex_omni_ros.core.engine.Engine`."""

    def __init__(self) -> None:
        self.sleeping = False

    def start(self) -> None:
        pass

    @property
    def started(self) -> bool:
        return True

    def sleep(self) -> None:
        self.sleeping = True

    def wake_up(self) -> None:
        self.sleeping = False

    def infer(self, request: InferenceRequest) -> InferenceResult:
        self.sleeping = False  # mirror the real engine's wake-on-infer
        start_time = time.monotonic()
        width, height = request.image.size
        raw_output = self._render_output(request)

        result = InferenceResult(raw_output=raw_output)
        if request.task is tasks.TaskType.KEYPOINT:
            result.keypoint_instances = parser.parse_keypoint_instances(
                raw_output, width, height
            )
            for instance in result.keypoint_instances:
                instance.confidence = 1.0
        else:
            result.annotations = parser.parse_annotations(raw_output, width, height)
            for annotation in result.annotations:
                annotation.confidence = 1.0
        result.inference_time = time.monotonic() - start_time
        return result

    def _render_output(self, request: InferenceRequest) -> str:
        task = request.task
        categories = request.categories or ["object"]

        if task is tasks.TaskType.KEYPOINT:
            names = tasks.KEYPOINT_SETS.get(request.keypoint_type, ["nose"])
            keypoints = {names[0]: " <250> <250> "}
            for name in names[1:]:
                keypoints[name] = "unvisible"
            instance = {"bbox": " <100> <100> <500> <500> ", "keypoints": keypoints}
            return "```json\n" + json.dumps({f"{categories[0]}1": instance}) + "\n```"

        if task in (tasks.TaskType.POINTING, tasks.TaskType.GUI_POINTING):
            coords = _POINT_TOKENS
        elif task is tasks.TaskType.OCR_POLYGON:
            coords = _POLYGON_TOKENS
        else:
            coords = _BOX_TOKENS

        if task is tasks.TaskType.VISUAL_PROMPTING:
            categories = ["object"]
        elif task in (tasks.TaskType.OCR_BOX, tasks.TaskType.OCR_POLYGON):
            categories = ["mock text"]

        return "".join(_ref_block(category, coords) for category in categories)
