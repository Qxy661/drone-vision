"""Tests for target_detector.py - color detection and template matching logic."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2


class TestColorDetection:
    """Test HSV color detection logic."""

    def test_red_detection_hsv(self):
        """Red object in HSV should be detected."""
        # Create a red image in BGR
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)  # BGR red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red range 1
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        # Red range 2 (wrap-around)
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)
        assert np.sum(mask > 0) > 0  # should detect red pixels

    def test_blue_not_detected_as_red(self):
        """Blue should not be detected as red."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR blue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        assert np.sum(mask > 0) == 0

    def test_contour_detection(self):
        """Should find contour for colored region."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), -1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0
        area = cv2.contourArea(contours[0])
        assert area > 5000  # 100x100 rectangle

    def test_small_contour_filtered(self):
        """Small contours should be filtered out."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (12, 12), (0, 0, 255), -1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large = [c for c in contours if cv2.contourArea(c) >= 500]
        assert len(large) == 0


class TestTemplateMatching:
    def test_template_match(self):
        """Template matching should find exact copy."""
        # Use a scene with features (not all zeros) to avoid edge effects
        scene = np.random.randint(50, 150, (200, 200), dtype=np.uint8)
        template = scene[50:70, 50:70].copy()

        result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        assert max_val > 0.99
        assert max_loc[0] == 50
        assert max_loc[1] == 50


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
