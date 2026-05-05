"""Kalman filter tracker unit tests"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from drone_vision.kalman_tracker import KalmanTracker, MultiTargetTracker


def test_static_target():
    """Stationary target: filter should converge to true position"""
    tracker = KalmanTracker(dt=0.033, measurement_noise=10.0)
    true_x, true_y = 320.0, 240.0

    # Simulate 30 noisy observations
    np.random.seed(42)
    for i in range(30):
        zx = true_x + np.random.normal(0, 5)
        zy = true_y + np.random.normal(0, 5)
        tracker.predict()
        tracker.update(zx, zy)

    ex, ey, _, _ = tracker.get_state()
    err = math.sqrt((ex - true_x)**2 + (ey - true_y)**2)
    assert err < 5.0, f"Error too large: {err}"
    print(f"  Static target: error={err:.2f}px, hits={tracker.hits}")


def test_moving_target():
    """Moving target: filter should track velocity"""
    tracker = KalmanTracker(dt=0.033)
    x, y = 100.0, 100.0
    vx_true, vy_true = 50.0, 30.0  # pixels/second

    for i in range(60):
        x += vx_true * 0.033
        y += vy_true * 0.033
        tracker.predict()
        tracker.update(x, y)

    ex, ey, evx, evy = tracker.get_state()
    vel_err = math.sqrt((evx - vx_true)**2 + (evy - vy_true)**2)
    assert vel_err < 10.0, f"Velocity error: {vel_err}"
    print(f"  Moving target: vel_error={vel_err:.2f} px/s")


def test_occlusion():
    """Target disappears for 5 frames, prediction should be reasonable"""
    tracker = KalmanTracker(dt=0.033)
    x, y = 200.0, 200.0

    # Track for 20 frames
    for i in range(20):
        x += 2.0
        tracker.predict()
        tracker.update(x, y)

    # Occlusion: 5 frames no observation
    for i in range(5):
        px, py = tracker.predict()
        tracker.mark_missed()

    # Position should be approximately where target went
    expected_x = x + 2.0 * 5
    ex, ey, _, _ = tracker.get_state()
    err = abs(ex - expected_x)
    assert err < 30, f"Occlusion prediction error: {err}"
    print(f"  Occlusion: pred_err={err:.1f}px, misses={tracker.misses}")


def test_multi_target():
    """Multi-target tracking with association"""
    mt = MultiTargetTracker(dt=0.033, max_missed=10)

    # Two targets moving in different directions
    t1_x, t1_y = 100.0, 100.0
    t2_x, t2_y = 400.0, 300.0

    for frame in range(30):
        t1_x += 3.0
        t2_y -= 2.0

        dets = [
            (t1_x, t1_y, 50, 50),
            (t2_x, t2_y, 60, 60),
        ]
        tracks = mt.update(dets)

    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    ids = [t['track_id'] for t in tracks]
    assert len(set(ids)) == 2, "Track IDs should be unique"
    print(f"  Multi-target: {len(tracks)} tracks, IDs={ids}")


if __name__ == "__main__":
    print("=== Kalman Tracker Tests ===")
    test_static_target()
    test_moving_target()
    test_occlusion()
    test_multi_target()
    print("=== ALL TESTS PASSED ===")
