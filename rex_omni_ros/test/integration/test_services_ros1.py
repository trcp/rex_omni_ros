"""Service round-trip tests against the node with the mock engine (ROS1).

Run via ``pixi run -e ros1 test-integration``; the task script starts roscore.
"""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from .util import make_bad_image_msg, make_image_msg

pytestmark = pytest.mark.skipif(
    os.environ.get("ROS_VERSION") != "1", reason="requires a ROS1 environment"
)

SERVICE_TIMEOUT = 30.0


@pytest.fixture(scope="module")
def services():
    import rospy

    with open("/tmp/rex_omni_test_node_ros1.log", "wb") as log:
        # start_new_session + killpg: kill the whole group in case rosrun
        # leaves the node as a separate child process.
        process = subprocess.Popen(
            ["rosrun", "rex_omni_ros", "rex_omni_server", "_backend:=mock"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rospy.init_node("rex_omni_test_client", anonymous=True)
        try:
            yield
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def proxy(srv_type, name):
    import rospy

    rospy.wait_for_service(name, timeout=SERVICE_TIMEOUT)
    return rospy.ServiceProxy(name, srv_type)


def test_detect(services):
    from rex_omni_msgs.srv import Detect, DetectRequest

    response = proxy(Detect, "/rex_omni/detect")(
        image=make_image_msg(),
        categories=["person", "dog"],
        variant=DetectRequest.VARIANT_DETECTION,
    )
    assert response.success, response.message
    assert [d.category for d in response.detections] == ["person", "dog"]
    detection = response.detections[0]
    assert 0 < detection.bbox.x0 < detection.bbox.x1 <= 640
    assert detection.confidence == pytest.approx(1.0)


def test_point(services):
    from rex_omni_msgs.srv import Point, PointRequest

    response = proxy(Point, "/rex_omni/point")(
        image=make_image_msg(),
        categories=["cup"],
        variant=PointRequest.VARIANT_POINTING,
    )
    assert response.success, response.message
    (point,) = response.points
    assert point.category == "cup"


def test_detect_with_visual_prompt(services):
    from rex_omni_msgs.msg import BoundingBox
    from rex_omni_msgs.srv import DetectWithVisualPrompt

    response = proxy(DetectWithVisualPrompt, "/rex_omni/detect_with_visual_prompt")(
        image=make_image_msg(),
        reference_boxes=[BoundingBox(x0=10.0, y0=10.0, x1=100.0, y1=100.0)],
    )
    assert response.success, response.message
    assert response.detections[0].category == "object"


def test_detect_keypoints(services):
    from rex_omni_msgs.srv import DetectKeypoints

    response = proxy(DetectKeypoints, "/rex_omni/detect_keypoints")(
        image=make_image_msg(), category="person", keypoint_type="person"
    )
    assert response.success, response.message
    (instance,) = response.instances
    assert len(instance.keypoints) == 17


def test_recognize_text_polygon(services):
    from rex_omni_msgs.srv import RecognizeText, RecognizeTextRequest

    response = proxy(RecognizeText, "/rex_omni/recognize_text")(
        image=make_image_msg(), variant=RecognizeTextRequest.VARIANT_POLYGON
    )
    assert response.success, response.message
    (region,) = response.regions
    assert region.text == "mock text"
    assert len(region.polygon.points) == 4


def test_unsupported_encoding_reports_error(services):
    from rex_omni_msgs.srv import Detect

    response = proxy(Detect, "/rex_omni/detect")(
        image=make_bad_image_msg(), categories=["person"], variant=0
    )
    assert not response.success
    assert "encoding" in response.message


def test_sleep_and_wake_up(services):
    from std_srvs.srv import Trigger

    response = proxy(Trigger, "/rex_omni/sleep")()
    assert response.success, response.message

    response = proxy(Trigger, "/rex_omni/wake_up")()
    assert response.success, response.message
