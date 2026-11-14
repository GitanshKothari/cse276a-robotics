"""Start HW3 hardware. Start the desired route separately once it is ready."""
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package, filename, **arguments):
    path = PathJoinSubstitution([FindPackageShare(package), "launch", filename])
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=arguments.items())


def generate_launch_description():
    return LaunchDescription([
        # apriltag_ros ignores lens distortion, so the camera must rectify.
        include("robot_vision_camera", "robot_vision_camera.launch.py",
                image_rectify="true"),
        include("apriltag_ros", "apriltag_launch.py", config_file="hw3_tags.yaml"),
        ExecuteProcess(cmd=[sys.executable, "-m", "robot.motor_control"], output="screen"),
    ])
