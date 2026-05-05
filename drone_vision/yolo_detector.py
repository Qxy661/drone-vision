"""
YOLOv8 目标检测节点
YOLOv8 Object Detection Node

替代原来简单的颜色检测, 使用 YOLOv8 做通用目标检测 + 卡尔曼滤波跟踪

优势:
- 通用检测: 能检测 80 类目标 (人、车、动物等)
- 实时性: YOLOv8n 在 Jetson Nano 上 ~15fps
- 鲁棒性: 对光照、尺度、角度变化鲁棒
- 卡尔曼跟踪: 平滑检测噪声, 处理遮挡

目标跟踪流程:
1. YOLOv8 检测当前帧的目标框
2. 卡尔曼滤波器预测每个跟踪目标的下一帧位置
3. 匹配检测框和跟踪器 (基于距离)
4. 更新匹配的跟踪器, 创建新的, 删除丢失的
5. 输出跟踪结果 (ID + 位置 + 速度)

适用场景:
- 无人机跟随地面车辆/人员
- 目标搜索和锁定
- 多目标监控
"""
import json
import time
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from drone_vision.kalman_tracker import MultiTargetTracker


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # 参数
        self.declare_parameter('video_source', 0)
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('target_class', -1)  # -1 = all classes
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('enable_tracking', True)

        src = self.get_parameter('video_source').value
        model_name = self.get_parameter('model_name').value
        self.target_class = self.get_parameter('target_class').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        self.enable_tracking = self.get_parameter('enable_tracking').value

        self.bridge = CvBridge()

        # 加载 YOLOv8
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.get_logger().info(f'YOLOv8 loaded: {model_name}')
        except ImportError:
            self.get_logger().warn(
                'ultralytics not installed, using fallback color detection')
            self.model = None

        # 卡尔曼多目标跟踪器
        self.tracker = MultiTargetTracker(
            dt=1.0/30.0, iou_threshold=0.3, max_missed=15)

        # 视频源
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open video: {src}')

        self.frame_count = 0

        # 发布
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.det_pub = self.create_publisher(
            String, '/vision/detections', 10)
        self.track_pub = self.create_publisher(
            String, '/vision/tracks', 10)
        self.img_pub = self.create_publisher(
            Image, '/vision/detection_image', qos)

        # 30fps
        self.timer = self.create_timer(1.0/30.0, self._process)

        self.get_logger().info(
            f'YOLODetector started (source={src}, model={model_name})')

    def _process(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1
        detections = []

        # YOLOv8 检测
        if self.model is not None:
            detections = self._detect_yolo(frame)
        else:
            detections = self._detect_color_fallback(frame)

        # 卡尔曼跟踪
        tracks = []
        if self.enable_tracking and detections:
            det_boxes = [
                (d['cx'], d['cy'], d.get('w', 50), d.get('h', 50))
                for d in detections
            ]
            tracks = self.tracker.update(det_boxes)

        # 发布检测结果
        det_msg = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'num_detections': len(detections),
            'detections': detections,
        }
        msg = String()
        msg.data = json.dumps(det_msg)
        self.det_pub.publish(msg)

        # 发布跟踪结果
        if tracks:
            track_msg = {
                'frame': self.frame_count,
                'num_tracks': len(tracks),
                'tracks': tracks,
            }
            tmsg = String()
            tmsg.data = json.dumps(track_msg)
            self.track_pub.publish(tmsg)

        # 发布标注图像
        if self.publish_annotated:
            annotated = self._annotate(frame, detections, tracks)
            img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            self.img_pub.publish(img_msg)

    def _detect_yolo(self, frame) -> list:
        """YOLOv8 检测"""
        results = self.model(frame, verbose=False, conf=self.conf_thresh)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 过滤目标类别
                if self.target_class >= 0 and cls_id != self.target_class:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                detections.append({
                    'cx': cx, 'cy': cy,
                    'w': w, 'h': h,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id],
                    'bbox': [x1, y1, x2, y2],
                })

        return detections

    def _detect_color_fallback(self, frame) -> list:
        """备用颜色检测 (YOLOv8 未安装时)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append({
                'cx': x + w/2, 'cy': y + h/2,
                'w': w, 'h': h,
                'confidence': 0.5,
                'class_name': 'color_target',
            })
        return detections

    def _annotate(self, frame, detections, tracks):
        annotated = frame.copy()

        # 画检测框
        for d in detections:
            cx, cy = int(d['cx']), int(d['cy'])
            w, h = int(d.get('w', 50)), int(d.get('h', 50))
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.get('class_name','?')} {d['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 画跟踪轨迹
        for t in tracks:
            tx, ty = int(t['x']), int(t['y'])
            vx, vy = t['vx'], t['vy']
            track_id = t['track_id']

            # 跟踪点
            cv2.circle(annotated, (tx, ty), 8, (0, 0, 255), -1)
            cv2.putText(annotated, f"ID:{track_id}",
                        (tx + 10, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            # 速度向量
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 1.0:
                end_x = int(tx + vx * 0.1)
                end_y = int(ty + vy * 0.1)
                cv2.arrowedLine(annotated, (tx, ty), (end_x, end_y),
                               (255, 0, 0), 2)

        # 信息
        info = f"Det: {len(detections)} | Trk: {len(tracks)} | F: {self.frame_count}"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
