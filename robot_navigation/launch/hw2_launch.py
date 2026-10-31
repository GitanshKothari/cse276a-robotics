"""Bring up the whole HW2 pipeline.

    camera -> apriltag detector -> localizer -> controller -> motor node

The camera node publishes rectified frames and matching camera_info, the
detector turns those into tag corners, the localizer folds them into a
pose, the controller closes the loop on that pose, and the motor node
drives the wheels. The logger records the run alongside.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = get_package_share_directory('robot_navigation')
    default_params = os.path.join(share, 'config', 'navigation.yaml')
    default_map = os.path.join(share, 'config', 'landmarks.yaml')
    default_apriltag = os.path.join(share, 'config', 'apriltag.yaml')

    arguments = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('map_file', default_value=default_map),
        DeclareLaunchArgument('apriltag_config', default_value=default_apriltag),
        DeclareLaunchArgument('output_dir', default_value=os.path.expanduser('~/hw2_runs')),
        DeclareLaunchArgument('run_name', default_value=''),
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='set false to replay a bag'),
        DeclareLaunchArgument('drive', default_value='true',
                              description='set false to localize without moving'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('use_cans', default_value='true',
                              description='run the coke can detector too'),
        # The can detector thresholds colour, so it needs a brighter and
        # more saturated picture than the tag detector does. The tag
        # detector reads a black and white pattern and does not care.
        # These feed GStreamer's videobalance: brightness -1..1,
        # contrast 0..2, saturation 0..2.
        # Drive speed. Lower both for the first run on a new landmark
        # layout: a slow rover gives the filter more sightings per metre,
        # and leaves you more time to reach the keyboard.
        DeclareLaunchArgument('max_linear', default_value='0.25'),
        DeclareLaunchArgument('max_angular', default_value='0.9'),
        # Frames per second. The sensor cannot expose for longer than one
        # frame interval, so in a dark room a lower rate is a brighter
        # picture. The detector needs only a few frames a second.
        DeclareLaunchArgument('frame_rate', default_value='5'),
        DeclareLaunchArgument('brightness', default_value='0.35'),
        DeclareLaunchArgument('contrast', default_value='1.0'),
        DeclareLaunchArgument('saturation', default_value='2.0'),
    ]

    params_file = LaunchConfiguration('params_file')

    camera = Node(
        package='robot_vision_camera',
        executable='robot_vision_camera_node',
        name='robot_vision_camera',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera')),
        parameters=[{
            'frame_rate': ParameterValue(
                LaunchConfiguration('frame_rate'), value_type=int),
            'width': 1280,
            'height': 720,
            'input_format': 'NV12',
            'output_format': 'BGR',
            'topic_name': 'camera/image_raw',
            'image_compress': True,
            # The detector does no undistortion of its own, so the frames
            # have to arrive already rectified.
            'image_rectify': True,
            'camera_parameter_path': PathJoinSubstitution([
                FindPackageShare('robot_vision_camera'),
                'config', 'camera_parameter.yaml',
            ]),
            'jpeg_quality': 80,
            'brightness': LaunchConfiguration('brightness'),
            'contrast': LaunchConfiguration('contrast'),
            'saturation': LaunchConfiguration('saturation'),
        }],
    )

    detector = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node',
        output='screen',
        remappings=[
            ('image_rect', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
        parameters=[LaunchConfiguration('apriltag_config')],
    )

    # The custom landmark option. The localizer takes tags and cans
    # through the same range and bearing update, so running both costs
    # nothing and either can be switched off on its own.
    cans = Node(
        package='robot_perception',
        executable='can_detector',
        name='can_detector',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_cans')),
        parameters=[PathJoinSubstitution([
            FindPackageShare('robot_perception'),
            'config', 'can_detector.yaml',
        ])],
    )

    localizer = Node(
        package='robot_navigation',
        executable='landmark_localizer',
        name='landmark_localizer',
        output='screen',
        parameters=[params_file, {'map_path': LaunchConfiguration('map_file')}],
    )

    motors = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
    )

    controller = Node(
        package='robot_navigation',
        executable='waypoint_controller',
        name='waypoint_controller',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('drive')),
        # The overrides come after the file, so they win.
        parameters=[params_file, {
            'max_linear': ParameterValue(
                LaunchConfiguration('max_linear'), value_type=float),
            'max_angular': ParameterValue(
                LaunchConfiguration('max_angular'), value_type=float),
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

    bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('foxglove')),
        parameters=[{'port': 8765, 'address': '0.0.0.0'}],
    )

    # If the motor node dies the rover is uncontrolled, so take the whole
    # stack down with it rather than leaving the last command standing.
    stop_on_motor_exit = RegisterEventHandler(OnProcessExit(
        target_action=motors,
        on_exit=[Shutdown(reason='motor controller exited')],
    ))

    return LaunchDescription(arguments + [
        camera, detector, cans, localizer, motors, controller, logger, bridge,
        stop_on_motor_exit,
    ])
