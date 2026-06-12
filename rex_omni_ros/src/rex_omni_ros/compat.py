"""ROS1/ROS2 compatibility layer.

Defines a single :class:`RosNode` class whose implementation is selected at
import time from the ``ROS_VERSION`` environment variable (set by the ROS /
RoboStack environment). The rest of the package interacts with ROS only
through this interface, so node code is identical for both versions.

Unified semantics:

* ``get_param(name, default)`` reads a private (``~``) parameter.
* ``create_service(srv_type, name, handler)`` registers a private service.
  ``handler`` always has the rospy-style signature ``handler(request) ->
  response`` and must return an instance of ``response_class(srv_type)``.
* ``create_publisher(msg_type, name)`` registers a private topic publisher
  and returns a :class:`Publisher`.
* ``spin()`` blocks until shutdown and releases resources afterwards.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Protocol, TypeVar, Union, cast

ParamValue = Union[bool, int, float, str, List[str]]
ParamT = TypeVar("ParamT", bound=ParamValue)
ServiceHandler = Callable[[Any], Any]


class Publisher(Protocol):
    """Minimal topic publisher shared by both ROS versions."""

    @property
    def subscriber_count(self) -> int: ...

    def publish(self, message: Any) -> None: ...

ROS_VERSION = int(os.environ.get("ROS_VERSION", "0"))


if ROS_VERSION == 1:
    import rospy

    class _RospyPublisher:
        def __init__(self, publisher: Any) -> None:
            self._publisher = publisher

        @property
        def subscriber_count(self) -> int:
            return int(self._publisher.get_num_connections())

        def publish(self, message: Any) -> None:
            self._publisher.publish(message)

    class RosNode:
        """rospy-backed implementation."""

        def __init__(self, name: str) -> None:
            rospy.init_node(name)
            self.name = name

        @staticmethod
        def request_class(srv_type: Any) -> Any:
            return srv_type._request_class

        @staticmethod
        def response_class(srv_type: Any) -> Any:
            return srv_type._response_class

        def get_param(self, name: str, default: ParamT) -> ParamT:
            return cast(ParamT, rospy.get_param("~" + name, default))

        def create_service(
            self, srv_type: Any, name: str, handler: ServiceHandler
        ) -> None:
            rospy.Service("~" + name, srv_type, handler)

        def create_publisher(self, msg_type: Any, name: str) -> Publisher:
            return _RospyPublisher(
                rospy.Publisher("~" + name, msg_type, queue_size=1)
            )

        def call_service(
            self, srv_type: Any, name: str, request: Any, timeout: float = 60.0
        ) -> Any:
            rospy.wait_for_service(name, timeout=timeout)
            return rospy.ServiceProxy(name, srv_type)(request)

        def log_info(self, message: str) -> None:
            rospy.loginfo(message)

        def log_warn(self, message: str) -> None:
            rospy.logwarn(message)

        def log_error(self, message: str) -> None:
            rospy.logerr(message)

        def spin(self) -> None:
            rospy.spin()

elif ROS_VERSION == 2:
    import rclpy
    import rclpy.node

    class _RclpyPublisher:
        def __init__(self, publisher: Any) -> None:
            self._publisher = publisher

        @property
        def subscriber_count(self) -> int:
            return int(self._publisher.get_subscription_count())

        def publish(self, message: Any) -> None:
            self._publisher.publish(message)

    class RosNode:  # type: ignore[no-redef]
        """rclpy-backed implementation."""

        def __init__(self, name: str) -> None:
            rclpy.init()
            self._node = rclpy.node.Node(name)
            self.name = name

        @staticmethod
        def request_class(srv_type: Any) -> Any:
            return srv_type.Request

        @staticmethod
        def response_class(srv_type: Any) -> Any:
            return srv_type.Response

        def get_param(self, name: str, default: ParamT) -> ParamT:
            return cast(ParamT, self._node.declare_parameter(name, default).value)

        def create_service(
            self, srv_type: Any, name: str, handler: ServiceHandler
        ) -> None:
            def callback(request: Any, _response: Any) -> Any:
                return handler(request)

            self._node.create_service(srv_type, "~/" + name, callback)

        def create_publisher(self, msg_type: Any, name: str) -> Publisher:
            return _RclpyPublisher(
                self._node.create_publisher(msg_type, "~/" + name, 1)
            )

        def call_service(
            self, srv_type: Any, name: str, request: Any, timeout: float = 60.0
        ) -> Any:
            client = self._node.create_client(srv_type, name)
            if not client.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f"service {name} not available")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
            response = future.result()
            if response is None:
                raise TimeoutError(f"call to {name} timed out")
            return response

        def log_info(self, message: str) -> None:
            self._node.get_logger().info(message)

        def log_warn(self, message: str) -> None:
            self._node.get_logger().warning(message)

        def log_error(self, message: str) -> None:
            self._node.get_logger().error(message)

        def spin(self) -> None:
            try:
                rclpy.spin(self._node)
            except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
                pass
            finally:
                self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()

else:
    raise ImportError(
        "ROS_VERSION environment variable must be '1' or '2'; is a ROS "
        "environment active?"
    )
