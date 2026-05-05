from setuptools import find_packages, setup

package_name = "drone_vision"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/vision_test.launch.py",
            "launch/vision_system.launch.py",
        ]),
        ("share/" + package_name + "/config", [
            "config/vision_params.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="dev@example.com",
    description="Drone visual target tracking - YOLOv8 + Kalman filter",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "target_detector = drone_vision.target_detector:main",
            "visual_servo = drone_vision.visual_servo:main",
            "tracking_manager = drone_vision.tracking_manager:main",
            "yolo_detector = drone_vision.yolo_detector:main",
        ],
    },
)
