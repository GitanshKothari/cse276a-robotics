"""Keyboard teleoperation. Publishes /motor_commands at 20 Hz.

W = forward, S = backward, A = left, D = right, X = stop, ESC = quit.
"""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# pynput is optional. Imported lazily inside the node so the rest of
# the package still works on systems where it is not installed.
Key = None
Listener = None


KEYS = {
    "w": (0.3, 0.3),
    "s": (-0.3, -0.3),
    "a": (-0.5, 0.5),
    "d": (0.5, -0.5),
    "x": (0.0, 0.0),
}


class KeyboardNode(Node):

    def __init__(self):
        super().__init__("keyboard_node")
        global Key, Listener
        from pynput.keyboard import Key as _K, Listener as _L
        Key, Listener = _K, _L
        self.pub = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )
        self.left = 0.0
        self.right = 0.0
        self.pressed = set()
        self.running = True
        self.last_logged = (None, None)

        threading.Thread(target=self._listen, daemon=True).start()
        self.create_timer(0.05, self._publish)   # 20 Hz
        self.get_logger().info(
            "Keyboard controller: W/A/S/D drive, X stop, ESC quit."
        )

    def _on_press(self, key):
        if not hasattr(key, "char") or not key.char:
            return
        c = key.char.lower()
        if c in KEYS:
            self.pressed.add(c)
            self.left, self.right = KEYS[c]
            if c == "x":
                self.pressed.clear()

    def _on_release(self, key):
        if key == Key.esc:
            self.running = False
            rclpy.shutdown()
            return False
        if hasattr(key, "char") and key.char:
            c = key.char.lower()
            self.pressed.discard(c)
            if not any(k in self.pressed for k in "wasd"):
                self.left = self.right = 0.0

    def _listen(self):
        with Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=True,
        ) as listener:
            listener.join()

    def _publish(self):
        if not self.running:
            return
        msg = Float32MultiArray()
        msg.data = [self.left, self.right]
        self.pub.publish(msg)
        now = (self.left, self.right)
        if now != self.last_logged:
            self.get_logger().info(f"Motor cmd L={self.left:.2f} R={self.right:.2f}")
            self.last_logged = now

    def destroy_node(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.pub.publish(msg)
        super().destroy_node()


def main():
    rclpy.init()
    node = KeyboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
