"""Tests for visual_servo.py - PIDController and servo logic."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from drone_vision.visual_servo import PIDController


class TestPIDController:
    def test_proportional(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=10.0)
        pid.prev_time = time.time() - 0.02
        out = pid.update(error=5.0, dt=0.02)
        assert abs(out - 5.0) < 0.1

    def test_integral(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limit=10.0)
        pid.prev_time = time.time() - 0.02
        # Accumulate integral over 5 steps
        for _ in range(5):
            out = pid.update(error=1.0, dt=0.1)
        # integral = 1.0 * 0.1 * 5 = 0.5
        assert abs(out - 0.5) < 0.01

    def test_derivative(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0, output_limit=10.0)
        pid.prev_time = time.time() - 0.02
        pid.update(error=0.0, dt=0.02)
        out = pid.update(error=1.0, dt=0.02)
        # derivative = (1.0 - 0.0) / 0.02 = 50
        assert abs(out - 50.0) > 0  # should be clamped to limit
        assert abs(out) <= 10.0 + 0.01

    def test_output_limit(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
        pid.prev_time = time.time() - 0.02
        out = pid.update(error=10.0, dt=0.02)
        assert abs(out - 1.0) < 0.01

    def test_negative_limit(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
        pid.prev_time = time.time() - 0.02
        out = pid.update(error=-10.0, dt=0.02)
        assert abs(out - (-1.0)) < 0.01

    def test_integral_anti_windup(self):
        """Integral should be clamped to prevent windup."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limit=1.0)
        pid.prev_time = time.time() - 0.02
        # Large error for many steps
        for _ in range(100):
            pid.update(error=100.0, dt=0.02)
        # Integral should be clamped to output_limit
        assert abs(pid.integral) <= pid.limit + 0.01

    def test_reset(self):
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        pid.prev_time = time.time() - 0.02
        pid.update(error=5.0, dt=0.02)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0

    def test_zero_error(self):
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0, output_limit=1.0)
        pid.prev_time = time.time() - 0.02
        out = pid.update(error=0.0, dt=0.02)
        assert abs(out) < 0.01


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
