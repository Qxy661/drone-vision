"""
视觉伺服控制 ROS2 节点
Visual Servoing Controller

将目标在图像中的位置转换为无人机速度命令
实现 Position-Based Visual Servoing (PBVS) 的简化版本

核心思想:
  目标在图像中心 -> 无人机不需要移动
  目标偏离中心 -> 无人机向目标方向移动
  偏离越大 -> 速度越快 (PID控制)

跟踪模式:
- CENTER: 保持目标在画面中心 (默认)
- FOLLOW: 跟随目标保持固定距离
- CIRCLE: 绕目标盘旋
"""
import json
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode


class ServoState:
    IDLE = "idle"
    ACQUIRING = "acquiring"     # 正在锁定目标
    TRACKING = "tracking"       # 跟踪中
    LOST = "lost"               # 目标丢失
    REACQUIRING = "reacquiring" # 重新搜索


class PIDController:
    """简单 PID 控制器
    u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt
    """
    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error, dt=None):
        now = time.time()
        if dt is None:
            dt = (now - self.prev_time) if self.prev_time else 0.02
        self.prev_time = now

        self.integral += error * dt
        # 抗积分饱和
        self.integral = max(-self.limit, min(self.limit, self.integral))

        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.limit, min(self.limit, output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class VisualServoNode(Node):
    def __init__(self):
        super().__init__("visual_servo")

        # 参数
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("tracking_mode", "center")  # center / follow / circle
        self.declare_parameter("max_velocity", 0.5)  # m/s
        self.declare_parameter("target_lost_timeout", 3.0)  # 秒
        self.declare_parameter("kp_xy", 0.002)
        self.declare_parameter("ki_xy", 0.0001)
        self.declare_parameter("kd_xy", 0.001)
        self.declare_parameter("test_mode", True)

        self.img_w = self.get_parameter("image_width").value
        self.img_h = self.get_parameter("image_height").value
        self.mode = self.get_parameter("tracking_mode").value
        self.max_vel = self.get_parameter("max_velocity").value
        self.lost_timeout = self.get_parameter("target_lost_timeout").value
        self.test_mode = self.get_parameter("test_mode").value

        # 状态
        self.servo_state = ServoState.IDLE
        self.target_detected = False
        self.target_cx = 0
        self.target_cy = 0
        self.target_area = 0
        self.last_detect_time = 0.0
        self.fcu_connected = False
        self.armed = False

        # PID 控制器 (x/y 方向)
        kp = self.get_parameter("kp_xy").value
        ki = self.get_parameter("ki_xy").value
        kd = self.get_parameter("kd_xy").value
        self.pid_x = PIDController(kp, ki, kd, self.max_vel)
        self.pid_y = PIDController(kp, ki, kd, self.max_vel)

        # 订阅
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(String, "/vision/detections",
                                 self._detection_cb, 10)
        if not self.test_mode:
            self.create_subscription(State, "/mavros/state",
                                     self._state_cb, 10)

        # 发布
        if not self.test_mode:
            self.vel_pub = self.create_publisher(
                TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self.status_pub = self.create_publisher(
            String, "/vision/servo_status", 10)

        # 模式切换
        if not self.test_mode:
            self.set_mode_client = self.create_client(
                SetMode, "/mavros/set_mode")

        # 控制循环 20Hz
        self.timer = self.create_timer(0.05, self._control_loop)

        mode_str = "TEST" if self.test_mode else "REAL"
        self.get_logger().info(
            f"VisualServo started (mode={self.mode}, {mode_str})")

    def _state_cb(self, msg):
        self.fcu_connected = msg.connected
        self.armed = msg.armed

    def _detection_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.target_detected = data.get("detected", False)
            if self.target_detected:
                self.target_cx = data.get("cx", self.img_w // 2)
                self.target_cy = data.get("cy", self.img_h // 2)
                self.target_area = data.get("area", 0)
                self.last_detect_time = time.time()

                if self.servo_state in (ServoState.IDLE, ServoState.LOST,
                                         ServoState.REACQUIRING):
                    self.servo_state = ServoState.ACQUIRING
                    self.pid_x.reset()
                    self.pid_y.reset()
        except json.JSONDecodeError:
            pass

    def _control_loop(self):
        now = time.time()

        # 检查目标是否丢失
        if self.servo_state == ServoState.TRACKING:
            if not self.target_detected:
                if now - self.last_detect_time > self.lost_timeout:
                    self.servo_state = ServoState.LOST
                    self.get_logger().warn("Target LOST")

        if self.servo_state == ServoState.ACQUIRING:
            if self.target_detected:
                self.servo_state = ServoState.TRACKING
                self.get_logger().info("Target acquired -> TRACKING")

        vx, vy, vz, vyaw = 0.0, 0.0, 0.0, 0.0

        if self.servo_state == ServoState.TRACKING and self.target_detected:
            # 计算目标偏离图像中心的误差 (像素)
            err_x = self.target_cx - self.img_w / 2.0   # 正值 = 目标在右边
            err_y = self.target_cy - self.img_h / 2.0   # 正值 = 目标在下方

            # PID 计算速度
            # x 方向: 图像 x 对应无人机的左右 (body frame y)
            # y 方向: 图像 y 对应无人机的前后 (body frame x, 因为相机朝下)
            vyaw = self.pid_x.update(err_x)    # 偏航修正
            vx = -self.pid_y.update(err_y)     # 前后修正 (图像y向下, 无人机向前)

            if self.mode == "follow":
                # FOLLOW: 根据目标面积调整前后距离
                target_area = 5000  # 期望面积
                area_err = self.target_area - target_area
                vx -= 0.0001 * area_err  # 面积大->后退, 面积小->前进

            elif self.mode == "circle":
                # CIRCLE: 绕目标盘旋
                vyaw = 0.3  # 固定角速度旋转

        elif self.servo_state == ServoState.LOST:
            # 目标丢失: 悬停或缓慢搜索旋转
            vyaw = 0.2  # 慢速旋转搜索

        # 发布速度命令
        if not self.test_mode:
            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.twist.linear.x = vx
            vel.twist.linear.y = vy
            vel.twist.linear.z = vz
            vel.twist.angular.z = vyaw
            self.vel_pub.publish(vel)
        else:
            if self.servo_state == ServoState.TRACKING:
                err_x = self.target_cx - self.img_w / 2.0
                err_y = self.target_cy - self.img_h / 2.0
                self.get_logger().info(
                    f"[SERVO] err=({err_x:.0f},{err_y:.0f}) "
                    f"vel=({vx:.3f},{vy:.3f},{vyaw:.3f})",
                    throttle_duration_sec=1.0)

        # 发布状态
        status = {
            "state": self.servo_state,
            "tracking_mode": self.mode,
            "target_detected": self.target_detected,
            "target_pos": [self.target_cx, self.target_cy],
            "velocity_cmd": [vx, vy, vz, vyaw],
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
