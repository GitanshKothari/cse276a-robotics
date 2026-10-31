from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='chl235@ucsd.edu',
    description='Closed-loop waypoint navigation with AprilTag landmark localization',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'landmark_localizer = robot_navigation.landmark_localizer:main',
            'waypoint_controller = robot_navigation.waypoint_controller:main',
            'trajectory_logger = robot_navigation.trajectory_logger:main',
            'plot_trajectory = robot_navigation.plot_trajectory:main',
            'check_coverage = robot_navigation.coverage_check:main',
            'sim_rover = robot_navigation.sim_rover:main',
        ],
    },
)
