"""
Visual Target Tracking - Test Mode
Uses webcam, no FCU connection
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory("drone_vision"), "config")
    params = os.path.join(config_dir, "vision_params.yaml")

    return LaunchDescription([
        Node(package="drone_vision", executable="target_detector",
             name="target_detector", output="screen",
             parameters=[params]),

        Node(package="drone_vision", executable="visual_servo",
             name="visual_servo", output="screen",
             parameters=[params, {"test_mode": True}]),

        Node(package="drone_vision", executable="tracking_manager",
             name="tracking_manager", output="screen",
             parameters=[params, {"test_mode": True}]),
    ])
