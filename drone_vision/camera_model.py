"""
相机模型与投影变换
Camera Model and Projection Utilities

针孔相机模型:
    u = fx * X/Z + cx
    v = fy * Y/Z + cy

其中:
    (X, Y, Z) = 3D点在相机坐标系中的坐标
    (u, v) = 像素坐标
    fx, fy = 焦距 (像素单位)
    (cx, cy) = 主点 (光心在图像上的投影)
"""
import math
import numpy as np
from typing import Tuple, Optional


class PinholeCamera:
    """针孔相机模型

    支持:
    - 像素坐标 <-> 归一化坐标转换
    - 3D点投影到像素
    - 像素反投影为射线
    - 简单畸变校正
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 distortion: Optional[list] = None,
                 image_width: int = 640, image_height: int = 480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = distortion or [0, 0, 0, 0, 0]
        self.width = image_width
        self.height = image_height

    def pixel_to_normalized(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 -> 归一化坐标 (去内参)
        归一化坐标 = (X/Z, Y/Z), 即3D点在z=1平面上的投影
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return x, y

    def normalized_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """归一化坐标 -> 像素坐标"""
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return u, v

    def pixel_to_ray(self, u: float, v: float) -> Tuple[float, float, float]:
        """像素坐标 -> 相机坐标系中的单位射线方向
        用于从2D图像点恢复3D方向信息
        """
        x, y = self.pixel_to_normalized(u, v)
        norm = math.sqrt(x*x + y*y + 1.0)
        return x / norm, y / norm, 1.0 / norm

    def project_point(self, X: float, Y: float, Z: float) -> Tuple[float, float]:
        """3D点投影到像素坐标
        (X, Y, Z) in camera frame -> (u, v) in image
        """
        if Z <= 0:
            return -1, -1  # behind camera
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        return u, v

    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """单点畸变校正 (简化版, 仅考虑径向畸变 k1, k2)
        """
        x, y = self.pixel_to_normalized(u, v)
        r2 = x*x + y*y
        k1, k2 = self.distortion[0], self.distortion[1]
        factor = 1 + k1 * r2 + k2 * r2 * r2
        x_corrected = x * factor
        y_corrected = y * factor
        return self.normalized_to_pixel(x_corrected, y_corrected)


def estimate_distance(pixel_size: float, known_real_size: float,
                      focal_length: float) -> float:
    """根据物体在图像中的像素大小估算距离
    公式: distance = (real_size * focal_length) / pixel_size

    适用场景: 已知目标实际大小, 通过图像中的像素大小推算距离
    精度: 取决于目标检测的精度和实际大小的准确性
    """
    if pixel_size <= 0:
        return float('inf')
    return (known_real_size * focal_length) / pixel_size


def pixel_offset_to_angle(u_offset: float, v_offset: float,
                           fx: float, fy: float) -> Tuple[float, float]:
    """像素偏移量 -> 角度偏移 (弧度)
    用于: 目标偏离图像中心的角度, 可用于控制无人机转向
    """
    yaw_angle = math.atan2(u_offset, fx)
    pitch_angle = math.atan2(v_offset, fy)
    return yaw_angle, pitch_angle
