import sys
import time
from pathlib import Path

import rclpy
from std_msgs.msg import Float32MultiArray

from .config import COMMAND_DT
from .trajectory_planner import load_waypoints, make_plan


# Name a wheel command for the printed plan.
def move_name(left, right):
    if left == 0.0 and right == 0.0:
        return "stop"
    if left < 0.0 and right > 0.0:
        return "left"
    if left > 0.0 and right < 0.0:
        return "right"
    return "straight"


# Print each move and the total time. A move is (seconds, left, right).
def print_plan(moves):
    t = 0.0
    for seconds, left, right in moves:
        print(
            f"t={t:5.2f}s  {seconds:5.2f}s  {move_name(left, right):8}  "
            f"L={left:+.2f}  R={right:+.2f}"
        )
        t += seconds
    print(f"Total duration: {t:.2f}s")


# Repeat one wheel command until its seconds are up.
# The motor node stops the rover if commands pause for more than 0.15 s.
def send(pub, seconds, left, right):
    end = time.monotonic() + seconds
    while time.monotonic() < end and rclpy.ok():
        pub.publish(Float32MultiArray(data=[left, right]))
        time.sleep(COMMAND_DT)


# Load waypoints, print the plan, then drive.
def main():
    path = Path(__file__).with_name("waypoints.txt")
    points = load_waypoints(path)
    moves = make_plan(points)
    print_plan(moves)

    rclpy.init()
    node = rclpy.create_node("trajectory_controller")
    pub = node.create_publisher(Float32MultiArray, "motor_commands", 10)

    try:
        for seconds, left, right in moves:
            send(pub, seconds, left, right)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            pub.publish(Float32MultiArray(data=[0.0, 0.0]))
            time.sleep(COMMAND_DT)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
