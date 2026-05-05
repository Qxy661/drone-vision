# drone-vision

ROS2 视觉目标跟踪系统。YOLOv8 检测 + 卡尔曼滤波跟踪 + PID 视觉伺服。

## 功能

- YOLOv8 通用目标检测 (支持任意 COCO 类别)
- 卡尔曼滤波多目标跟踪 (状态估计 + 遮挡预测)
- PID 视觉伺服 (CENTER/FOLLOW/CIRCLE 三模式)
- 针孔相机模型 (像素->射线, 投影, PnP)
- 颜色检测 + ArUco 检测 (备选方案)

## 架构

```
摄像头/RTSP -> yolo_detector -> kalman_tracker -> visual_servo -> MAVROS -> 飞控
              target_detector (备选)
```

## Modules

| Module | 可独立使用 | 功能 |
|--------|-----------|------|
| camera_model | Yes | 针孔相机模型 |
| kalman_tracker | Yes | 卡尔曼滤波 + 多目标跟踪 |
| target_detector | No (ROS) | OpenCV 目标检测 |
| yolo_detector | No (ROS) | YOLOv8 通用检测 |
| visual_servo | No (ROS) | PID 视觉伺服 |
| tracking_manager | No (ROS) | 跟踪任务编排 |

## 快速开始

```bash
# 安装依赖
pip install opencv-python numpy ultralytics
sudo apt install ros-humble-cv-bridge ros-humble-image-transport

# 编译
cd ros2_ws && colcon build --packages-select drone_vision
source install/setup.bash

# 运行测试
python3 src/drone_vision/test/test_camera_model.py
python3 src/drone_vision/test/test_kalman.py

# 启动系统
ros2 launch drone_vision vision_test.launch.py
```

## 视觉伺服模式

| 模式 | 功能 | 适用场景 |
|------|------|---------|
| CENTER | 保持目标在画面中心 | 固定点监控 |
| FOLLOW | 跟随目标移动 | 目标跟踪 |
| CIRCLE | 绕目标做圆周运动 | 环绕拍摄 |

## License

MIT
