"""Bring up the full robot stack:
    camera -> AprilTag detector -> motor node -> navigator.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("robot_vision_camera"),
                "launch", "robot_vision_camera.launch.py",
            ])
        ])
    )

    apriltag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("apriltag_ros"),
                "launch", "apriltag_launch.py",
            ])
        ])
    )

    motors = Node(
        package="robot", executable="motor_node",
        name="motor_node", output="screen",
    )

    navigator = Node(
        package="robot", executable="navigator",
        name="navigator", output="screen",
    )

    return LaunchDescription([
        camera,
        TimerAction(period=4.0, actions=[apriltag]),
        TimerAction(period=6.0, actions=[motors]),
        TimerAction(period=8.0, actions=[navigator]),
    ])
