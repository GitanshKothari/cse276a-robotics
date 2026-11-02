"""Motor node.

Subscribes to /motor_commands (Float32MultiArray [L, R]) and writes JSON
to the robot over serial. Reads IMU replies and publishes /imu_data.
Stops the wheels if no command arrives for MOTOR_SAFETY_TIMEOUT seconds.
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

from .config import (
    SERIAL_PORT, SERIAL_BAUD,
    MOTOR_SAFETY_TIMEOUT, MOTOR_TX_HZ, IMU_TX_HZ,
)


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y_, z


class _SerialLink:
    """Wraps the serial connection and handles reconnect on error."""

    def __init__(self, logger):
        self.log = logger
        self.port = None
        self.open()

    def open(self):
        try:
            self.port = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
            self.log.info(f"Serial open: {SERIAL_PORT}")
        except Exception as e:
            self.log.warn(f"Serial open failed: {e}")
            self.port = None

    def write(self, obj):
        if self.port is None:
            return
        try:
            self.port.write((json.dumps(obj) + "\n").encode("utf-8"))
        except Exception as e:
            self.log.error(f"Serial write failed: {e}")
            self.open()

    def read_lines(self):
        """Yield complete JSON lines from the serial buffer."""
        if self.port is None:
            return
        buf = ""
        try:
            data = self.port.read(1024)
            if data:
                buf = data.decode("utf-8", errors="ignore")
        except Exception as e:
            self.log.error(f"Serial read failed: {e}")
            return
        for line in buf.split("\n"):
            line = line.strip()
            if line:
                yield line

    def close(self):
        if self.port:
            try:
                self.port.close()
            except Exception:
                pass


class MotorNode(Node):
    """ROS node that bridges /motor_commands and /imu_data to the robot."""

    def __init__(self):
        super().__init__("motor_node")
        self.link = _SerialLink(self.get_logger())

        self.create_subscription(
            Float32MultiArray, "motor_commands",
            self._on_motor_command, 10,
        )
        self.imu_pub = self.create_publisher(Imu, "imu_data", 10)

        self.cmd = {"T": 1, "L": 0.0, "R": 0.0}
        self.cmd_lock = threading.Lock()
        self.last_cmd_time = time.time()
        self.timed_out = False
        self.running = True

        threading.Thread(target=self._tx_motor_loop, daemon=True).start()
        threading.Thread(target=self._tx_imu_loop, daemon=True).start()
        threading.Thread(target=self._rx_loop, daemon=True).start()
        self.get_logger().info(
            f"Motor node up. Safety timeout {MOTOR_SAFETY_TIMEOUT:.2f}s"
        )

    # ---------- callbacks ----------

    def _on_motor_command(self, msg):
        if len(msg.data) < 2:
            self.get_logger().warn("Motor command needs 2 values [L, R]")
            return
        L = float(np.clip(msg.data[0], -0.5, 0.5))
        R = float(np.clip(msg.data[1], -0.5, 0.5))
        with self.cmd_lock:
            self.cmd = {"T": 1, "L": L, "R": R}
            self.last_cmd_time = time.time()
            if self.timed_out:
                self.get_logger().info("Motor commands resumed.")
                self.timed_out = False

    # ---------- loops ----------

    def _tx_motor_loop(self):
        period = 1.0 / MOTOR_TX_HZ
        while self.running and rclpy.ok():
            with self.cmd_lock:
                if (time.time() - self.last_cmd_time
                        > MOTOR_SAFETY_TIMEOUT):
                    if not self.timed_out:
                        self.get_logger().warn(
                            "Safety stop: no motor commands."
                        )
                        self.timed_out = True
                    self.cmd = {"T": 1, "L": 0.0, "R": 0.0}
                current = dict(self.cmd)
            self.link.write(current)
            time.sleep(period)

    def _tx_imu_loop(self):
        period = 1.0 / IMU_TX_HZ
        request = {"T": 126}
        while self.running and rclpy.ok():
            self.link.write(request)
            time.sleep(period)

    def _rx_loop(self):
        while self.running and rclpy.ok():
            for line in self.link.read_lines():
                self._handle_rx(line)
            time.sleep(0.01)

    def _handle_rx(self, line):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict) and data.get("T") == 1002:
            self._publish_imu(data)

    def _publish_imu(self, d):
        try:
            w, x, y, z = euler_to_quaternion(
                float(d.get("r", 0)),
                float(d.get("p", 0)),
                float(d.get("y", 0)),
            )
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "imu_link"
            msg.orientation.w = w
            msg.orientation.x = x
            msg.orientation.y = y
            msg.orientation.z = z
            msg.angular_velocity.x = float(d.get("gx", 0))
            msg.angular_velocity.y = float(d.get("gy", 0))
            msg.angular_velocity.z = float(d.get("gz", 0))
            msg.linear_acceleration.x = float(d.get("ax", 0)) * 9.81 / 1000.0
            msg.linear_acceleration.y = float(d.get("ay", 0)) * 9.81 / 1000.0
            msg.linear_acceleration.z = float(d.get("az", 0)) * 9.81 / 1000.0
            self.imu_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"IMU publish failed: {e}")

    def destroy_node(self):
        self.running = False
        try:
            self.link.write({"T": 1, "L": 0.0, "R": 0.0})
        except Exception:
            pass
        self.link.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
