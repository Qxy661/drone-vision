"""Camera model unit tests"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drone_vision.camera_model import (
    PinholeCamera, estimate_distance, pixel_offset_to_angle
)
import math


def test_pixel_normalized_roundtrip():
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
    u, v = 400, 300
    x, y = cam.pixel_to_normalized(u, v)
    u2, v2 = cam.normalized_to_pixel(x, y)
    assert abs(u2 - u) < 0.01
    assert abs(v2 - v) < 0.01
    print("  pixel<->normalized roundtrip: OK")


def test_project_point():
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
    # Point directly ahead at z=2m
    u, v = cam.project_point(0, 0, 2.0)
    assert abs(u - 320) < 0.01  # should be at center
    assert abs(v - 240) < 0.01
    print(f"  project (0,0,2) -> ({u:.0f},{v:.0f}): OK")


def test_pixel_to_ray():
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
    rx, ry, rz = cam.pixel_to_ray(320, 240)  # center pixel
    assert abs(rz - 1.0) < 0.01  # ray should point along z
    print(f"  center pixel ray: ({rx:.3f},{ry:.3f},{rz:.3f}): OK")


def test_estimate_distance():
    # Object 0.5m wide, focal 500px, appears 250px wide
    d = estimate_distance(250, 0.5, 500)
    assert abs(d - 1.0) < 0.01  # should be 1m away
    print(f"  distance estimate: {d:.2f}m: OK")


def test_pixel_offset_to_angle():
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
    yaw, pitch = pixel_offset_to_angle(0, 0, 500, 500)
    assert abs(yaw) < 0.01  # center = no angle
    print(f"  center offset angle: ({yaw:.3f},{pitch:.3f})rad: OK")


if __name__ == "__main__":
    print("=== Camera Model Tests ===")
    test_pixel_normalized_roundtrip()
    test_project_point()
    test_pixel_to_ray()
    test_estimate_distance()
    test_pixel_offset_to_angle()
    print("=== ALL TESTS PASSED ===")
