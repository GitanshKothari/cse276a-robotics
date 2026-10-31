"""Run the HW2 stack against the simulated rover instead of the hardware.

    sim_rover -> localizer -> controller -> sim_rover

Same localizer, same controller, same parameters as hw2_launch; only the
camera, detector and motor node are replaced by sim_rover, which closes
the loop by integrating the wheel commands and synthesising detections
from the map. Use it to check the loop converges and the frames are right
before taking the rover onto the floor.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = get_package_share_directory('robot_navigation')
    default_params = os.path.join(share, 'config', 'navigation.yaml')
    default_map = os.path.join(share, 'config', 'landmarks.yaml')

    arguments = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('map_file', default_value=default_map),
        DeclareLaunchArgument('output_dir',
                              default_value=os.path.expanduser('~/hw2_runs')),
        DeclareLaunchArgument('run_name', default_value=''),
        DeclareLaunchArgument('seed', default_value='1'),
        DeclareLaunchArgument('pixel_sigma', default_value='0.7'),
        DeclareLaunchArgument('dropout_probability', default_value='0.0',
                              description='chance of losing each tag per frame'),
        DeclareLaunchArgument('shutdown_after_complete', default_value='1.5',
                              description='seconds held stopped before the run ends'),
    ]

    params_file = LaunchConfiguration('params_file')
    map_file = LaunchConfiguration('map_file')

    rover = Node(
        package='robot_navigation',
        executable='sim_rover',
        name='sim_rover',
        output='screen',
        parameters=[params_file, {
            'map_path': map_file,
            'camera_config': PathJoinSubstitution([
                FindPackageShare('robot_vision_camera'),
                'config', 'camera_parameter.yaml',
            ]),
            'seed': LaunchConfiguration('seed'),
            'pixel_sigma': LaunchConfiguration('pixel_sigma'),
            'dropout_probability': LaunchConfiguration('dropout_probability'),
        }],
    )

    localizer = Node(
        package='robot_navigation',
        executable='landmark_localizer',
        name='landmark_localizer',
        output='screen',
        parameters=[params_file, {'map_path': map_file}],
    )

    controller = Node(
        package='robot_navigation',
        executable='waypoint_controller',
        name='waypoint_controller',
        output='screen',
        emulate_tty=True,
        parameters=[params_file, {
            'shutdown_after_complete': LaunchConfiguration('shutdown_after_complete'),
        }],
    )

    logger = Node(
        package='robot_navigation',
        executable='trajectory_logger',
        name='trajectory_logger',
        output='screen',
        parameters=[params_file, {
            'output_dir': LaunchConfiguration('output_dir'),
            'run_name': LaunchConfiguration('run_name'),
        }],
    )

    # The controller exits once the route is finished, which is the cue to
    # tear the run down so a scripted sweep of seeds can move on.
    stop_when_done = RegisterEventHandler(OnProcessExit(
        target_action=controller,
        on_exit=[Shutdown(reason='route complete')],
    ))

    return LaunchDescription(arguments + [
        rover, localizer, controller, logger,
        stop_when_done,
    ])
