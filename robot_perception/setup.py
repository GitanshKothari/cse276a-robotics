from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='chl235@ucsd.edu',
    description='Coke can landmark detection for HW2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'can_detector = robot_perception.can_detector_node:main',
            'can_check = robot_perception.can_check:main',
        ],
    },
)
