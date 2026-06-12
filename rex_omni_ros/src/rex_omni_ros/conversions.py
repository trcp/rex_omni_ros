"""Conversions between ROS messages and core types.

The generated message classes of rex_omni_msgs share module paths and field
names between ROS1 and ROS2, so these helpers work unchanged for both.
sensor_msgs/Image is decoded with numpy directly to avoid a cv_bridge/OpenCV
dependency.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image as PILImage

from rex_omni_ros.core import types

# encoding -> (number of channels, channel order mapping to RGB(A))
_SUPPORTED_ENCODINGS: dict[str, tuple[int, list[int] | None]] = {
    "rgb8": (3, None),
    "bgr8": (3, [2, 1, 0]),
    "rgba8": (4, None),
    "bgra8": (4, [2, 1, 0, 3]),
    "mono8": (1, None),
}


def image_msg_to_pil(msg: Any) -> PILImage.Image:
    """Decode a sensor_msgs/Image into a PIL image.

    Raises:
        ValueError: If the encoding is unsupported or the buffer is malformed.
    """
    encoding = msg.encoding.lower()
    if encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; "
            f"expected one of {sorted(_SUPPORTED_ENCODINGS)}"
        )
    channels, channel_order = _SUPPORTED_ENCODINGS[encoding]

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    row_bytes = msg.width * channels
    step = msg.step if msg.step >= row_bytes else row_bytes
    if data.size < msg.height * step:
        raise ValueError(
            f"image buffer too small: got {data.size} bytes, "
            f"expected at least {msg.height * step}"
        )
    array = data[: msg.height * step].reshape(msg.height, step)[:, :row_bytes]
    array = array.reshape(msg.height, msg.width, channels)
    if channel_order is not None:
        array = array[:, :, channel_order]
    if channels == 1:
        array = array[:, :, 0]
    return PILImage.fromarray(np.ascontiguousarray(array))


def compressed_image_msg_to_pil(msg: Any) -> PILImage.Image:
    """Decode a sensor_msgs/CompressedImage (jpeg, png, ...) into a PIL image.

    image_transport's compressed plugin encodes the source array without
    channel reordering and records it in the format string (e.g.
    ``"bgr8; jpeg compressed bgr8"``); BGR(A) data is swapped back to RGB(A).

    Raises:
        ValueError: If the payload is not a decodable image.
    """
    image: PILImage.Image
    try:
        image = PILImage.open(io.BytesIO(bytes(msg.data)))
        image.load()
    except Exception as error:
        raise ValueError(
            f"failed to decode compressed image (format {msg.format!r}): {error}"
        ) from error
    if "compressed bgr" in msg.format.lower():
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] >= 3:
            channel_order = [2, 1, 0] + list(range(3, array.shape[2]))
            image = PILImage.fromarray(array[:, :, channel_order])
    return image


def decode_image_msg(msg: Any) -> PILImage.Image:
    """Decode a sensor_msgs Image or CompressedImage into a PIL image."""
    if hasattr(msg, "encoding"):
        return image_msg_to_pil(msg)
    return compressed_image_msg_to_pil(msg)


def pil_to_image_msg(image: PILImage.Image) -> Any:
    """Encode a PIL image as an rgb8 sensor_msgs/Image."""
    from sensor_msgs.msg import Image

    rgb = image.convert("RGB")
    msg = Image()
    msg.height = rgb.height
    msg.width = rgb.width
    msg.encoding = "rgb8"
    msg.step = rgb.width * 3
    msg.data = rgb.tobytes()
    return msg


def box_msg_to_core(msg: Any) -> types.Box:
    return types.Box(x0=msg.x0, y0=msg.y0, x1=msg.x1, y1=msg.y1)


def _box_msg(box: types.Box, msg_module: Any) -> Any:
    return msg_module.BoundingBox(
        x0=float(box.x0), y0=float(box.y0), x1=float(box.x1), y1=float(box.y1)
    )


def _point_msg(point: types.Point, msg_module: Any) -> Any:
    return msg_module.Point2D(x=float(point.x), y=float(point.y))


def annotation_to_detection_msg(annotation: types.Annotation, msg_module: Any) -> Any:
    assert isinstance(annotation.shape, types.Box)
    return msg_module.Detection(
        category=annotation.category,
        bbox=_box_msg(annotation.shape, msg_module),
        confidence=float(annotation.confidence),
    )


def annotation_to_point_msg(annotation: types.Annotation, msg_module: Any) -> Any:
    assert isinstance(annotation.shape, types.Point)
    return msg_module.PointDetection(
        category=annotation.category,
        point=_point_msg(annotation.shape, msg_module),
        confidence=float(annotation.confidence),
    )


def annotation_to_text_region_msg(annotation: types.Annotation, msg_module: Any) -> Any:
    region = msg_module.TextRegion(
        text=annotation.category, confidence=float(annotation.confidence)
    )
    if isinstance(annotation.shape, types.Box):
        region.bbox = _box_msg(annotation.shape, msg_module)
    elif isinstance(annotation.shape, types.Polygon):
        region.polygon = msg_module.Polygon2D(
            points=[_point_msg(p, msg_module) for p in annotation.shape.points]
        )
    else:  # a point is degenerate for OCR but preserved as a 1-vertex polygon
        region.polygon = msg_module.Polygon2D(
            points=[_point_msg(annotation.shape, msg_module)]
        )
    return region


def keypoint_instance_to_msg(instance: types.KeypointInstance, msg_module: Any) -> Any:
    keypoints = []
    for keypoint in instance.keypoints:
        visible = keypoint.position is not None
        position = (
            _point_msg(keypoint.position, msg_module)
            if keypoint.position is not None
            else msg_module.Point2D(x=0.0, y=0.0)
        )
        keypoints.append(
            msg_module.Keypoint(name=keypoint.name, position=position, visible=visible)
        )
    return msg_module.KeypointInstance(
        category=instance.category,
        bbox=_box_msg(instance.box, msg_module),
        keypoints=keypoints,
        confidence=float(instance.confidence),
    )
