"""Rex-Omni service node, common to ROS1 and ROS2."""

from __future__ import annotations

import functools
from typing import Any

from PIL import Image as PILImage
from std_srvs.srv import Trigger

from rex_omni_msgs.srv import (
    Detect,
    DetectKeypoints,
    DetectWithVisualPrompt,
    Point,
    RecognizeText,
)
from rex_omni_ros import conversions
from rex_omni_ros.compat import RosNode
from rex_omni_ros.core import visualization
from rex_omni_ros.core.engine import (
    Engine,
    EngineConfig,
    InferenceResult,
    RexOmniEngine,
)
from rex_omni_ros.core.mock import MockEngine
from rex_omni_ros.handlers import ResultCallback, RexOmniHandlers


def _load_config(node: RosNode) -> EngineConfig:
    defaults = EngineConfig()
    return EngineConfig(
        model_path=str(node.get_param("model_path", defaults.model_path)),
        gpu_memory_utilization=float(
            node.get_param("gpu_memory_utilization", defaults.gpu_memory_utilization)
        ),
        max_model_len=int(node.get_param("max_model_len", defaults.max_model_len)),
        min_pixels=int(node.get_param("min_pixels", defaults.min_pixels)),
        max_pixels=int(node.get_param("max_pixels", defaults.max_pixels)),
        max_tokens=int(node.get_param("max_tokens", defaults.max_tokens)),
        temperature=float(node.get_param("temperature", defaults.temperature)),
        top_p=float(node.get_param("top_p", defaults.top_p)),
        top_k=int(node.get_param("top_k", defaults.top_k)),
        repetition_penalty=float(
            node.get_param("repetition_penalty", defaults.repetition_penalty)
        ),
        system_prompt=str(node.get_param("system_prompt", defaults.system_prompt)),
        enable_confidence=bool(
            node.get_param("enable_confidence", defaults.enable_confidence)
        ),
        quantization=str(node.get_param("quantization", defaults.quantization)),
        dtype=str(node.get_param("dtype", defaults.dtype)),
        enforce_eager=bool(node.get_param("enforce_eager", defaults.enforce_eager)),
        warmup=bool(node.get_param("warmup", defaults.warmup)),
        compile_vit=bool(node.get_param("compile_vit", defaults.compile_vit)),
        enable_sleep_mode=bool(
            node.get_param("enable_sleep_mode", defaults.enable_sleep_mode)
        ),
    )


def _create_engine(node: RosNode, config: EngineConfig) -> Engine:
    backend = str(node.get_param("backend", "vllm"))
    if backend == "mock":
        node.log_warn("using mock engine; responses are canned test data")
        return MockEngine()
    if backend != "vllm":
        raise ValueError(f"unknown backend {backend!r}; expected 'vllm' or 'mock'")
    return RexOmniEngine(config)


def _make_debug_image_publisher(node: RosNode) -> ResultCallback | None:
    """A result callback publishing annotated images on ``~debug_image``.

    Rendering is skipped while the topic has no subscribers, so the default-on
    publisher costs nothing in normal operation.
    """
    if not bool(node.get_param("publish_debug_image", True)):
        return None
    from sensor_msgs.msg import Image

    publisher = node.create_publisher(Image, "debug_image")

    def publish(image: PILImage.Image, result: InferenceResult, header: Any) -> None:
        if publisher.subscriber_count == 0:
            return
        message = conversions.pil_to_image_msg(
            visualization.render_result(image, result)
        )
        message.header = header
        publisher.publish(message)

    return publish


def main() -> None:
    node = RosNode("rex_omni")
    config = _load_config(node)
    engine = _create_engine(node, config)

    node.log_info(f"loading model {config.model_path} (this may take a while)...")
    engine.start()

    handlers = RexOmniHandlers(
        engine,
        log_error=node.log_error,
        on_result=_make_debug_image_publisher(node),
    )
    services = [
        (Detect, "detect", handlers.handle_detect),
        (Point, "point", handlers.handle_point),
        (
            DetectWithVisualPrompt,
            "detect_with_visual_prompt",
            handlers.handle_detect_with_visual_prompt,
        ),
        (DetectKeypoints, "detect_keypoints", handlers.handle_detect_keypoints),
        (RecognizeText, "recognize_text", handlers.handle_recognize_text),
        (Trigger, "sleep", handlers.handle_sleep),
        (Trigger, "wake_up", handlers.handle_wake_up),
    ]
    for srv_type, name, handler in services:
        node.create_service(
            srv_type,
            name,
            functools.partial(handler, response_cls=RosNode.response_class(srv_type)),
        )

    node.log_info(
        "rex_omni ready; services: " + ", ".join(name for _, name, _ in services)
    )
    node.spin()


if __name__ == "__main__":
    main()
