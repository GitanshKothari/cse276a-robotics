import os
from glob import glob

from setuptools import setup

package_name = "robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "pyserial>=3.5",
        "pynput>=1.7.0",
        "numpy>=1.20.0",
    ],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@localhost",
    description="Single-package robot stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigator  = robot.navigator:main",
            "motor_node = robot.motor_control:main",
            "keyboard   = robot.keyboard:main",
            "calibrate  = robot.calibrate:main",
            "hw3_run    = robot.hw3_run:main",
        ],
    },
)
