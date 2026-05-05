"""
跟踪管理器 ROS2 节点
High-level tracking mission orchestration

状态机: IDLE -> SEARCHING -> TRACKING -> RETURNING -> IDLE
"""
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from mavros_msgs.srv import SetMode


class TrackState:
    IDLE = "idle"
    SEARCHING = "searching"
    TRACKING = "tracking"
    RETURNING = "returning"


class TrackingManagerNode(Node):
    def __init__(self):
        super().__init__("tracking_manager")

        self.declare_parameter("test_mode", True)
        self.declare_parameter("search_timeout", 30.0)
        self.declare_parameter("max_track_time", 120.0)
        self.declare_parameter("max_altitude", 5.0)
        self.declare_parameter("max_speed", 1.0)

        self.test_mode = self.get_parameter("test_mode").value
        self.search_timeout = self.get_parameter("search_timeout").value
        self.max_track_time = self.get_parameter("max_track_time").value

        self.state = TrackState.IDLE
        self.servo_state = "idle"
        self.start_time = 0.0
        self.track_start = 0.0
        self.rtl_sent = False

        # 订阅
        self.create_subscription(String, "/vision/servo_status",
                                 self._servo_cb, 10)
        self.create_subscription(String, "/vision/mission_cmd",
                                 self._cmd_cb, 10)

        # 发布
        self.status_pub = self.create_publisher(
            String, "/vision/mission_status", 10)

        # 模式切换
        if not self.test_mode:
            self.set_mode_client = self.create_client(
                SetMode, "/mavros/set_mode")

        self.timer = self.create_timer(0.5, self._update)
        self.get_logger().info(
            f"TrackingManager started (test_mode={self.test_mode})")

    def _servo_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.servo_state = data.get("state", "idle")
        except json.JSONDecodeError:
            pass

    def _cmd_cb(self, msg):
        """接收任务命令: {"cmd": "start"/"stop"}"""
        try:
            data = json.loads(msg.data)
            cmd = data.get("cmd")
            if cmd == "start" and self.state == TrackState.IDLE:
                self.state = TrackState.SEARCHING
                self.start_time = time.time()
                self.get_logger().info("Mission started -> SEARCHING")
            elif cmd == "stop":
                self.state = TrackState.IDLE
                self.get_logger().info("Mission stopped")
        except json.JSONDecodeError:
            pass

    def _update(self):
        now = time.time()

        if self.state == TrackState.SEARCHING:
            if self.servo_state == "tracking":
                self.state = TrackState.TRACKING
                self.track_start = now
                self.get_logger().info("Target found -> TRACKING")
            elif now - self.start_time > self.search_timeout:
                self.get_logger().warn("Search timeout -> returning")
                self.state = TrackState.RETURNING

        elif self.state == TrackState.TRACKING:
            if self.servo_state == "lost":
                self.get_logger().warn("Target lost during track -> searching")
                self.state = TrackState.SEARCHING
                self.start_time = now
            elif now - self.track_start > self.max_track_time:
                self.get_logger().info("Max track time -> returning")
                self.state = TrackState.RETURNING

        elif self.state == TrackState.RETURNING:
            if not self.rtl_sent:
                if not self.test_mode:
                    self._set_mode("RTL")
                self.rtl_sent = True
                self.get_logger().info("Returning to launch")
            # Stay in RETURNING until disarmed or manual stop
            if self.test_mode:
                self.state = TrackState.IDLE
                self.rtl_sent = False

        self._publish_status()

    def _set_mode(self, mode):
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            return
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_client.call_async(req)

    def _publish_status(self):
        status = {
            "state": self.state,
            "servo_state": self.servo_state,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
