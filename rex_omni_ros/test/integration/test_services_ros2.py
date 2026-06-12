"""Service round-trip tests against the node with the mock engine (ROS2).

Run via ``pixi run -e ros2 test-integration``; requires the workspace to be
built and sourced.
"""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from .util import (
    FRAME_ID,
    make_bad_image_msg,
    make_compressed_image_msg,
    make_image_msg,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ROS_VERSION") != "2", reason="requires a ROS2 environment"
)

SERVICE_TIMEOUT = 30.0


@pytest.fixture(scope="module")
def client_node():
    import rclpy

    with open("/tmp/rex_omni_test_node_ros2.log", "wb") as log:
        # start_new_session + killpg: `ros2 run` is a wrapper; terminating it
        # alone orphans the actual node process.
        process = subprocess.Popen(
            [
                "ros2",
                "run",
                "rex_omni_ros",
                "rex_omni_server",
                "--ros-args",
                "-p",
                "backend:=mock",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rclpy.init()
        node = rclpy.create_node("rex_omni_test_client")
        try:
            yield node
        finally:
            node.destroy_node()
            rclpy.shutdown()
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def call(node, srv_type, name, request):
    import rclpy

    client = node.create_client(srv_type, name)
    assert client.wait_for_service(timeout_sec=SERVICE_TIMEOUT), (
        f"service {name} not available"
    )
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=SERVICE_TIMEOUT)
    response = future.result()
    assert response is not None, f"call to {name} timed out"
    return response


def test_detect(client_node):
    from rex_omni_msgs.srv import Detect

    request = Detect.Request(
        image=make_image_msg(),
        categories=["person", "dog"],
        variant=Detect.Request.VARIANT_DETECTION,
    )
    response = call(client_node, Detect, "/rex_omni/detect", request)
    assert response.success, response.message
    assert response.header.frame_id == FRAME_ID
    assert [d.category for d in response.detections] == ["person", "dog"]
    detection = response.detections[0]
    assert 0 < detection.bbox.x0 < detection.bbox.x1 <= 640
    assert 0 < detection.bbox.y0 < detection.bbox.y1 <= 480
    assert detection.confidence == pytest.approx(1.0)
    assert response.inference_time >= 0.0


def test_detect_with_compressed_image(client_node):
    from rex_omni_msgs.srv import Detect

    request = Detect.Request(
        compressed_image=make_compressed_image_msg(), categories=["person"]
    )
    response = call(client_node, Detect, "/rex_omni/detect", request)
    assert response.success, response.message
    assert response.header.frame_id == FRAME_ID
    assert [d.category for d in response.detections] == ["person"]


def test_detect_without_any_image_reports_error(client_node):
    from rex_omni_msgs.srv import Detect

    request = Detect.Request(categories=["person"])
    response = call(client_node, Detect, "/rex_omni/detect", request)
    assert not response.success
    assert "no image data" in response.message


def test_point(client_node):
    from rex_omni_msgs.srv import Point

    request = Point.Request(
        image=make_image_msg(),
        categories=["cup"],
        variant=Point.Request.VARIANT_POINTING,
    )
    response = call(client_node, Point, "/rex_omni/point", request)
    assert response.success, response.message
    (point,) = response.points
    assert point.category == "cup"
    assert 0 < point.point.x <= 640 and 0 < point.point.y <= 480


def test_detect_with_visual_prompt(client_node):
    from rex_omni_msgs.msg import BoundingBox
    from rex_omni_msgs.srv import DetectWithVisualPrompt

    request = DetectWithVisualPrompt.Request(
        image=make_image_msg(),
        reference_boxes=[BoundingBox(x0=10.0, y0=10.0, x1=100.0, y1=100.0)],
    )
    response = call(
        client_node,
        DetectWithVisualPrompt,
        "/rex_omni/detect_with_visual_prompt",
        request,
    )
    assert response.success, response.message
    assert len(response.detections) == 1
    assert response.detections[0].category == "object"


def test_detect_keypoints(client_node):
    from rex_omni_msgs.srv import DetectKeypoints

    request = DetectKeypoints.Request(
        image=make_image_msg(), category="person", keypoint_type="person"
    )
    response = call(client_node, DetectKeypoints, "/rex_omni/detect_keypoints", request)
    assert response.success, response.message
    (instance,) = response.instances
    assert instance.category == "person"
    assert len(instance.keypoints) == 17
    visible = [kp for kp in instance.keypoints if kp.visible]
    assert [kp.name for kp in visible] == ["nose"]


def test_recognize_text_box(client_node):
    from rex_omni_msgs.srv import RecognizeText

    request = RecognizeText.Request(
        image=make_image_msg(), variant=RecognizeText.Request.VARIANT_WORD_BOX
    )
    response = call(client_node, RecognizeText, "/rex_omni/recognize_text", request)
    assert response.success, response.message
    (region,) = response.regions
    assert region.text == "mock text"
    assert region.bbox.x1 > region.bbox.x0


def test_recognize_text_polygon(client_node):
    from rex_omni_msgs.srv import RecognizeText

    request = RecognizeText.Request(
        image=make_image_msg(), variant=RecognizeText.Request.VARIANT_POLYGON
    )
    response = call(client_node, RecognizeText, "/rex_omni/recognize_text", request)
    assert response.success, response.message
    (region,) = response.regions
    assert len(region.polygon.points) == 4


def test_unsupported_encoding_reports_error(client_node):
    from rex_omni_msgs.srv import Detect

    request = Detect.Request(image=make_bad_image_msg(), categories=["person"])
    response = call(client_node, Detect, "/rex_omni/detect", request)
    assert not response.success
    assert "encoding" in response.message


def test_sleep_and_wake_up(client_node):
    from std_srvs.srv import Trigger

    response = call(client_node, Trigger, "/rex_omni/sleep", Trigger.Request())
    assert response.success, response.message

    response = call(client_node, Trigger, "/rex_omni/wake_up", Trigger.Request())
    assert response.success, response.message
