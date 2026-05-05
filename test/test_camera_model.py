"""Camera model unit tests"""
import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from drone_vision.camera_model import (
    PinholeCamera, estimate_distance, pixel_offset_to_angle
)


class TestPinholeCamera(unittest.TestCase):
    def test_pixel_normalized_roundtrip(self):
        cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
        u, v = 400, 300
        x, y = cam.pixel_to_normalized(u, v)
        u2, v2 = cam.normalized_to_pixel(x, y)
        assert abs(u2 - u) < 0.01
        assert abs(v2 - v) < 0.01

    def test_project_point(self):
        cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
        u, v = cam.project_point(0, 0, 2.0)
        assert abs(u - 320) < 0.01
        assert abs(v - 240) < 0.01

    def test_pixel_to_ray(self):
        cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
        rx, ry, rz = cam.pixel_to_ray(320, 240)
        assert abs(rz - 1.0) < 0.01

    def test_estimate_distance(self):
        d = estimate_distance(250, 0.5, 500)
        assert abs(d - 1.0) < 0.01

    def test_pixel_offset_to_angle(self):
        cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
        yaw, pitch = pixel_offset_to_angle(0, 0, 500, 500)
        assert abs(yaw) < 0.01


if __name__ == "__main__":
    unittest.main()
