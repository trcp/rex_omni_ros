"""Rendering of inference results onto images, for debug visualization."""

from __future__ import annotations

from typing import Union

from PIL import Image, ImageDraw, ImageFont

from rex_omni_ros.core.engine import InferenceResult
from rex_omni_ros.core.types import Box, Point, Polygon

# Distinct colors cycled over categories (Set1-like palette).
PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#17becf",
]


Font = Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]


def _default_font(line_width: int) -> Font:
    try:
        return ImageFont.load_default(size=max(12, 6 * line_width))
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    anchor: tuple[float, float],
    color: str,
    font: Font,
    line_width: int,
) -> None:
    """Draw a filled label box with its top-left near ``anchor``."""
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_height = bottom - top
    x = max(0.0, anchor[0])
    y = anchor[1] - text_height - 2 * line_width
    if y < 0:  # keep the label inside the image at the top edge
        y = anchor[1]
    draw.rectangle(
        (x, y, x + (right - left) + 2 * line_width, y + text_height + 2 * line_width),
        fill=color,
    )
    draw.text((x + line_width, y + line_width), label, fill="white", font=font)


def render_result(image: Image.Image, result: InferenceResult) -> Image.Image:
    """Draw all predictions of ``result`` on an RGB copy of ``image``."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 300))
    font = _default_font(line_width)
    radius = 2 * line_width
    colors: dict[str, str] = {}

    def color_for(category: str) -> str:
        return colors.setdefault(category, PALETTE[len(colors) % len(PALETTE)])

    for annotation in result.annotations:
        color = color_for(annotation.category)
        label = f"{annotation.category} {annotation.confidence:.2f}"
        shape = annotation.shape
        if isinstance(shape, Box):
            draw.rectangle(
                (shape.x0, shape.y0, shape.x1, shape.y1),
                outline=color,
                width=line_width,
            )
            _draw_label(draw, label, (shape.x0, shape.y0), color, font, line_width)
        elif isinstance(shape, Point):
            draw.ellipse(
                (shape.x - radius, shape.y - radius, shape.x + radius, shape.y + radius),
                fill=color,
            )
            _draw_label(draw, label, (shape.x, shape.y), color, font, line_width)
        elif isinstance(shape, Polygon) and shape.points:
            outline = [(p.x, p.y) for p in shape.points]
            draw.line(outline + outline[:1], fill=color, width=line_width)
            _draw_label(draw, label, outline[0], color, font, line_width)

    for instance in result.keypoint_instances:
        color = color_for(instance.category)
        box = instance.box
        draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=color, width=line_width)
        _draw_label(
            draw,
            f"{instance.category} {instance.confidence:.2f}",
            (box.x0, box.y0),
            color,
            font,
            line_width,
        )
        for keypoint in instance.keypoints:
            if keypoint.position is None:
                continue
            position = keypoint.position
            draw.ellipse(
                (
                    position.x - radius,
                    position.y - radius,
                    position.x + radius,
                    position.y + radius,
                ),
                fill=color,
                outline="white",
            )

    return canvas
