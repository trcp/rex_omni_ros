"""Service handlers: ROS request -> core inference -> ROS response.

Handlers are independent of the ROS runtime (rospy/rclpy); they only touch
generated message classes, which are identical in module path and field names
for both ROS versions. Each handler takes the request and the response class
(resolved by the compat layer) and returns a filled response.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from rex_omni_msgs import msg as msg_module
from rex_omni_ros import conversions
from rex_omni_ros.core import types
from rex_omni_ros.core.engine import Engine, InferenceRequest, InferenceResult
from rex_omni_ros.core.tasks import (
    OCR_TEXTLINE_CATEGORY,
    OCR_WORD_CATEGORY,
    TaskType,
)

LogFunction = Callable[[str], None]

_DETECT_VARIANTS = {0: TaskType.DETECTION, 1: TaskType.GUI_GROUNDING}
_POINT_VARIANTS = {0: TaskType.POINTING, 1: TaskType.GUI_POINTING}
# RecognizeText variant -> (task, categories)
_TEXT_VARIANTS = {
    0: (TaskType.OCR_BOX, [OCR_WORD_CATEGORY]),
    1: (TaskType.OCR_BOX, [OCR_TEXTLINE_CATEGORY]),
    2: (TaskType.OCR_POLYGON, [OCR_TEXTLINE_CATEGORY]),
}


class RexOmniHandlers:
    """Maps the five rex_omni_msgs services onto an :class:`Engine`."""

    def __init__(self, engine: Engine, log_error: LogFunction | None = None) -> None:
        self._engine = engine
        self._lock = threading.Lock()  # vLLM is not safe for concurrent calls
        self._log_error = log_error or (lambda message: None)

    @staticmethod
    def _select_image_msg(request: Any) -> Any:
        """The image message carrying the request's pixels (raw wins)."""
        if len(request.image.data) > 0:
            return request.image
        if len(request.compressed_image.data) > 0:
            return request.compressed_image
        raise ValueError(
            "request contains no image data; set image or compressed_image"
        )

    def _run(
        self,
        response: Any,
        request: Any,
        task: TaskType,
        categories: list[str] | None = None,
        keypoint_type: str = "",
        visual_prompt_boxes: list[types.Box] | None = None,
    ) -> InferenceResult | None:
        """Run inference, filling error fields on failure."""
        try:
            image_msg = self._select_image_msg(request)
            response.header = image_msg.header
            image = conversions.decode_image_msg(image_msg)
            with self._lock:
                result = self._engine.infer(
                    InferenceRequest(
                        image=image,
                        task=task,
                        categories=categories or [],
                        keypoint_type=keypoint_type,
                        visual_prompt_boxes=visual_prompt_boxes or [],
                    )
                )
        except Exception as error:  # noqa: BLE001 - reported via the response
            self._log_error(f"{task.value} request failed: {error}")
            response.success = False
            response.message = str(error)
            return None
        response.success = True
        response.inference_time = float(result.inference_time)
        return result

    def _switch_power_state(
        self, response_cls: Any, action: Callable[[], None], success_message: str
    ) -> Any:
        """Run an engine sleep/wake transition as a std_srvs/Trigger call."""
        response = response_cls()
        try:
            with self._lock:
                action()
        except Exception as error:  # noqa: BLE001 - reported via the response
            self._log_error(f"{action.__name__} request failed: {error}")
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = success_message
        return response

    def handle_sleep(self, request: Any, response_cls: Any) -> Any:
        return self._switch_power_state(
            response_cls, self._engine.sleep, "model offloaded to host RAM"
        )

    def handle_wake_up(self, request: Any, response_cls: Any) -> Any:
        return self._switch_power_state(
            response_cls, self._engine.wake_up, "model restored to VRAM"
        )

    def handle_detect(self, request: Any, response_cls: Any) -> Any:
        response = response_cls()
        task = _DETECT_VARIANTS.get(request.variant)
        if task is None:
            response.message = f"unknown variant {request.variant}"
            return response
        result = self._run(
            response, request, task, categories=list(request.categories)
        )
        if result is not None:
            response.detections = [
                conversions.annotation_to_detection_msg(a, msg_module)
                for a in result.annotations
                if isinstance(a.shape, types.Box)
            ]
        return response

    def handle_point(self, request: Any, response_cls: Any) -> Any:
        response = response_cls()
        task = _POINT_VARIANTS.get(request.variant)
        if task is None:
            response.message = f"unknown variant {request.variant}"
            return response
        result = self._run(
            response, request, task, categories=list(request.categories)
        )
        if result is not None:
            response.points = [
                conversions.annotation_to_point_msg(a, msg_module)
                for a in result.annotations
                if isinstance(a.shape, types.Point)
            ]
        return response

    def handle_detect_with_visual_prompt(self, request: Any, response_cls: Any) -> Any:
        response = response_cls()
        boxes = [conversions.box_msg_to_core(b) for b in request.reference_boxes]
        result = self._run(
            response,
            request,
            TaskType.VISUAL_PROMPTING,
            visual_prompt_boxes=boxes,
        )
        if result is not None:
            response.detections = [
                conversions.annotation_to_detection_msg(a, msg_module)
                for a in result.annotations
                if isinstance(a.shape, types.Box)
            ]
        return response

    def handle_detect_keypoints(self, request: Any, response_cls: Any) -> Any:
        response = response_cls()
        result = self._run(
            response,
            request,
            TaskType.KEYPOINT,
            categories=[request.category],
            keypoint_type=request.keypoint_type,
        )
        if result is not None:
            response.instances = [
                conversions.keypoint_instance_to_msg(instance, msg_module)
                for instance in result.keypoint_instances
            ]
        return response

    def handle_recognize_text(self, request: Any, response_cls: Any) -> Any:
        response = response_cls()
        variant = _TEXT_VARIANTS.get(request.variant)
        if variant is None:
            response.message = f"unknown variant {request.variant}"
            return response
        task, categories = variant
        result = self._run(response, request, task, categories=categories)
        if result is not None:
            response.regions = [
                conversions.annotation_to_text_region_msg(a, msg_module)
                for a in result.annotations
            ]
        return response
