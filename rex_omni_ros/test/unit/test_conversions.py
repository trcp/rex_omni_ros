"""Tests for the ROS-message-free part of conversions (image decoding).

The decoders only read attributes off the message objects, so plain
SimpleNamespace stand-ins are enough; no ROS environment is required.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rex_omni_ros.conversions import (
    compressed_image_msg_to_pil,
    decode_image_msg,
    image_msg_to_pil,
)


def make_raw_msg(array: np.ndarray, encoding: str) -> SimpleNamespace:
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        step=width * channels,
        data=array.tobytes(),
    )


def make_compressed_msg(image: Image.Image, format: str = "png") -> SimpleNamespace:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return SimpleNamespace(format=format, data=buffer.getvalue())


def gradient_image(width: int = 32, height: int = 16) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :] * 7
    array[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 13
    array[:, :, 2] = 200
    return Image.fromarray(array)


def test_raw_rgb8_roundtrip() -> None:
    expected = gradient_image()

    decoded = image_msg_to_pil(make_raw_msg(np.asarray(expected), "rgb8"))

    assert np.array_equal(np.asarray(decoded), np.asarray(expected))


def test_raw_bgr8_channels_swapped() -> None:
    expected = gradient_image()
    bgr = np.asarray(expected)[:, :, ::-1]

    decoded = image_msg_to_pil(make_raw_msg(np.ascontiguousarray(bgr), "bgr8"))

    assert np.array_equal(np.asarray(decoded), np.asarray(expected))


def test_compressed_png_roundtrip() -> None:
    expected = gradient_image()

    decoded = compressed_image_msg_to_pil(make_compressed_msg(expected, "png"))

    assert np.array_equal(
        np.asarray(decoded.convert("RGB")), np.asarray(expected)
    )


def test_compressed_bgr_format_swaps_channels_back() -> None:
    # image_transport stores the BGR array as-is and marks it in the format.
    expected = gradient_image()
    stored = Image.fromarray(np.asarray(expected)[:, :, ::-1])

    decoded = compressed_image_msg_to_pil(
        make_compressed_msg(stored, "bgr8; png compressed bgr8")
    )

    assert np.array_equal(np.asarray(decoded), np.asarray(expected))


def test_compressed_rejects_garbage() -> None:
    msg = SimpleNamespace(format="jpeg", data=b"not an image")

    with pytest.raises(ValueError, match="failed to decode"):
        compressed_image_msg_to_pil(msg)


def test_decode_dispatches_on_message_shape() -> None:
    expected = gradient_image()

    raw = decode_image_msg(make_raw_msg(np.asarray(expected), "rgb8"))
    compressed = decode_image_msg(make_compressed_msg(expected))

    assert np.array_equal(np.asarray(raw), np.asarray(expected))
    assert np.array_equal(np.asarray(compressed.convert("RGB")), np.asarray(expected))
