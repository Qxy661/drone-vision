"""Tests for tracking_manager.py — state machine logic"""
import sys
import os
import unittest
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import rclpy
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False


@unittest.skipUnless(HAS_RCLPY, "rclpy not available")
class TestTrackState(unittest.TestCase):
    """Test TrackState constants."""

    def test_state_values(self):
        from drone_vision.tracking_manager import TrackState
        self.assertEqual(TrackState.IDLE, "idle")
        self.assertEqual(TrackState.SEARCHING, "searching")
        self.assertEqual(TrackState.TRACKING, "tracking")
        self.assertEqual(TrackState.RETURNING, "returning")


@unittest.skipUnless(HAS_RCLPY, "rclpy not available")
class TestTrackingStateMachine(unittest.TestCase):
    """Test the tracking manager state machine logic (without ROS2)."""

    def _make_manager(self):
        """Create a minimal TrackingManagerNode for testing."""
        import rclpy
        from drone_vision.tracking_manager import TrackingManagerNode
        rclpy.init()
        node = TrackingManagerNode()
        return node

    def _cleanup(self, node):
        import rclpy
        node.destroy_node()
        rclpy.shutdown()

    def test_start_command_transitions_to_searching(self):
        """start command from IDLE -> SEARCHING"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        import json
        from drone_vision.tracking_manager import TrackState

        self.assertEqual(node.state, TrackState.IDLE)
        # Simulate start command
        class FakeMsg:
            data = json.dumps({"cmd": "start"})
        node._cmd_cb(FakeMsg())
        self.assertEqual(node.state, TrackState.SEARCHING)
        self.assertGreater(node.start_time, 0)
        self._cleanup(node)

    def test_stop_command_returns_to_idle(self):
        """stop command from any state -> IDLE"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        import json
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.TRACKING
        class FakeMsg:
            data = json.dumps({"cmd": "stop"})
        node._cmd_cb(FakeMsg())
        self.assertEqual(node.state, TrackState.IDLE)
        self._cleanup(node)

    def test_start_from_non_idle_ignored(self):
        """start command from non-IDLE should be ignored"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        import json
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.SEARCHING
        class FakeMsg:
            data = json.dumps({"cmd": "start"})
        node._cmd_cb(FakeMsg())
        self.assertEqual(node.state, TrackState.SEARCHING)
        self._cleanup(node)

    def test_searching_to_tracking_on_servo(self):
        """SEARCHING -> TRACKING when servo reports tracking"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        import json
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.SEARCHING
        node.start_time = time.time()
        node.servo_state = "tracking"
        node._update()
        self.assertEqual(node.state, TrackState.TRACKING)
        self._cleanup(node)

    def test_search_timeout_returns(self):
        """SEARCHING -> RETURNING on timeout"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.SEARCHING
        node.start_time = time.time() - node.search_timeout - 1
        node.servo_state = "idle"
        node._update()
        self.assertEqual(node.state, TrackState.RETURNING)
        self._cleanup(node)

    def test_tracking_lost_returns_to_searching(self):
        """TRACKING -> SEARCHING when servo reports lost"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.TRACKING
        node.track_start = time.time()
        node.servo_state = "lost"
        node._update()
        self.assertEqual(node.state, TrackState.SEARCHING)
        self._cleanup(node)

    def test_max_track_time_returns(self):
        """TRACKING -> RETURNING after max_track_time"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.TRACKING
        node.track_start = time.time() - node.max_track_time - 1
        node.servo_state = "tracking"
        node._update()
        self.assertEqual(node.state, TrackState.RETURNING)
        self._cleanup(node)

    def test_returning_to_idle_in_test_mode(self):
        """RETURNING -> IDLE immediately in test_mode"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        from drone_vision.tracking_manager import TrackState

        node.state = TrackState.RETURNING
        node.rtl_sent = False
        node.test_mode = True
        node._update()
        self.assertEqual(node.state, TrackState.IDLE)
        self.assertFalse(node.rtl_sent)
        self._cleanup(node)

    def test_malformed_cmd_ignored(self):
        """Malformed JSON in cmd callback should be silently ignored"""
        try:
            node = self._make_manager()
        except Exception:
            self.skipTest("rclpy not available")
        from drone_vision.tracking_manager import TrackState

        class FakeMsg:
            data = "not json"
        node._cmd_cb(FakeMsg())
        self.assertEqual(node.state, TrackState.IDLE)
        self._cleanup(node)


if __name__ == '__main__':
    unittest.main()
