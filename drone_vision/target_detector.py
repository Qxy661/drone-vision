"""
目标检测 ROS2 节点
Target Detector Node

检测方法:
1. 颜色检测 (HSV) - 适用于颜色鲜明的目标
2. ArUco 标记检测 - 适用于预设标记
3. 模板匹配 - 适用于已知外形的目标

订阅: 摄像头图像
发布: 检测结果 JSON + 标注后图像
"""
import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class TargetDetectorNode(Node):
    def __init__(self):
        super().__init__("target_detector")

        # 参数
        self.declare_parameter("detection_method", "color")  # color / aruco / template
        self.declare_parameter("video_source", 0)
        # HSV 范围 (默认检测红色)
        self.declare_parameter("hsv_lower_h", 0)
        self.declare_parameter("hsv_lower_s", 120)
        self.declare_parameter("hsv_lower_v", 70)
        self.declare_parameter("hsv_upper_h", 10)
        self.declare_parameter("hsv_upper_s", 255)
        self.declare_parameter("hsv_upper_v", 255)
        self.declare_parameter("min_area", 500)
        self.declare_parameter("max_area", 50000)
        self.declare_parameter("publish_annotated", True)

        self.method = self.get_parameter("detection_method").value
        self.video_source = self.get_parameter("video_source").value
        self.hsv_lower = np.array([
            self.get_parameter("hsv_lower_h").value,
            self.get_parameter("hsv_lower_s").value,
            self.get_parameter("hsv_lower_v").value,
        ])
        self.hsv_upper = np.array([
            self.get_parameter("hsv_upper_h").value,
            self.get_parameter("hsv_upper_s").value,
            self.get_parameter("hsv_upper_v").value,
        ])
        self.min_area = self.get_parameter("min_area").value
        self.max_area = self.get_parameter("max_area").value
        self.publish_annotated = self.get_parameter("publish_annotated").value

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_detection = None

        # 视频源
        src = self.video_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open video source: {self.video_source}")

        # ArUco 检测器
        if self.method == "aruco":
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()

        # 发布
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.det_pub = self.create_publisher(String, "/vision/detections", 10)
        self.img_pub = self.create_publisher(Image, "/vision/detection_image", qos)

        # 定时器 30fps
        self.timer = self.create_timer(1.0 / 30.0, self._process_frame)

        self.get_logger().info(
            f"TargetDetector started (method={self.method}, source={self.video_source})")

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1
        detection = None

        if self.method == "color":
            detection = self._detect_color(frame)
        elif self.method == "aruco":
            detection = self._detect_aruco(frame)
        elif self.method == "template":
            detection = self._detect_template(frame)

        self.last_detection = detection

        # 发布检测结果
        result = {
            "detected": detection is not None,
            "frame": self.frame_count,
            "method": self.method,
            "timestamp": time.time(),
        }
        if detection:
            result.update(detection)

        msg = String()
        msg.data = json.dumps(result)
        self.det_pub.publish(msg)

        # 发布标注图像
        if self.publish_annotated:
            annotated = self._annotate(frame, detection)
            img_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            self.img_pub.publish(img_msg)

    def _detect_color(self, frame):
        """颜色检测: HSV 颜色空间分割 + 轮廓分析"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area or area > self.max_area:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(largest)

        return {
            "cx": cx, "cy": cy,
            "area": area,
            "bbox": [x, y, w, h],
            "confidence": min(area / 5000.0, 1.0),
        }

    def _detect_aruco(self, frame):
        """ArUco 标记检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None

        # 取第一个检测到的标记
        corner = corners[0][0]
        cx = int(corner[:, 0].mean())
        cy = int(corner[:, 1].mean())
        area = cv2.contourArea(corner.astype(np.float32))

        return {
            "cx": cx, "cy": cy,
            "area": area,
            "marker_id": int(ids[0][0]),
            "corners": corner.tolist(),
            "confidence": 0.95,
        }

    def _detect_template(self, frame):
        """模板匹配 (需要预加载模板图像)"""
        # 简化实现: 使用边缘检测 + 轮廓
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        return {
            "cx": int(M["m10"] / M["m00"]),
            "cy": int(M["m01"] / M["m00"]),
            "area": area,
            "confidence": 0.5,
        }

    def _annotate(self, frame, detection):
        """在图像上绘制检测结果"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        # 画中心十字线
        cv2.line(annotated, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 1)
        cv2.line(annotated, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 1)

        if detection:
            cx, cy = detection["cx"], detection["cy"]
            # 画目标位置
            cv2.circle(annotated, (cx, cy), 10, (0, 0, 255), 2)
            cv2.line(annotated, (w//2, h//2), (cx, cy), (0, 0, 255), 1)
            # bbox
            if "bbox" in detection:
                x, y, bw, bh = detection["bbox"]
                cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
            # 信息文字
            info = f"dist: ({cx-w//2}, {cy-h//2}) area: {detection['area']:.0f}"
            cv2.putText(annotated, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "NO TARGET", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
