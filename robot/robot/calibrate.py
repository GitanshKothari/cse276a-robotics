"""Camera calibration (checkerboard, 9x6 inner corners, 24 mm squares).

Two modes:
  python -m robot.calibrate collect   # subscribe /camera/image_raw and
                                      # save frames when ENTER is pressed
  python -m robot.calibrate solve     # run OpenCV calibration on saved
                                      # frames and write camera_info.yaml
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.024          # metres
BLUR_THRESHOLD = 35.0
TARGET_IMAGES = 40
MIN_IMAGES = 10
CAMERA_TOPIC = "/camera/image_raw"

ROOT = Path("/home/ubuntu/ros2_ws/camera_calibration")
IMG_DIR = ROOT / "images"
PRE_DIR = ROOT / "previews"
YAML_PATH = ROOT / "camera_info.yaml"


# -------- helpers --------

def _detect_corners(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if ok:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ok, corners, gray


def _ros_to_bgr(msg):
    enc = msg.encoding.lower()
    if enc not in ("rgb8", "bgr8", "mono8"):
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "mono8":
        img = np.ndarray(
            (msg.height, msg.width), dtype=np.uint8,
            buffer=buf, strides=(msg.step, 1),
        ).copy()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ndarray(
        (msg.height, msg.width, 3), dtype=np.uint8,
        buffer=buf, strides=(msg.step, 3, 1),
    ).copy()
    if enc == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _next_id():
    ids = {
        int(p.stem.split("_")[-1])
        for d, pat in [(IMG_DIR, "frame_*.png"), (PRE_DIR, "corners_*.png")]
        for p in d.glob(pat) if p.stem.split("_")[-1].isdigit()
    }
    return max(ids, default=0) + 1


# -------- mode: collect --------

class _Collector:

    def __init__(self):
        self.node = rclpy.create_node("checkerboard_collector")
        self.frame = None
        self.stamp = None
        self.lock = threading.Lock()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.node.create_subscription(Image, CAMERA_TOPIC, self._cb, qos)

    def _cb(self, msg):
        try:
            img = _ros_to_bgr(msg)
            with self.lock:
                self.frame = img
                self.stamp = time.monotonic()
        except Exception as e:
            print("convert failed:", e)

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.stamp


def collect():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PRE_DIR.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    coll = _Collector()
    stop = threading.Event()
    threading.Thread(
        target=lambda: [
            rclpy.spin_once(coll.node, timeout_sec=0.1)
            for _ in iter(lambda: not stop.is_set() and rclpy.ok(), False)
        ],
        daemon=True,
    ).start()
    try:
        count = len(list(IMG_DIR.glob("frame_*.png")))
        print(f"Have {count}/{TARGET_IMAGES} frames. Press ENTER to capture, q to quit.")
        while count < TARGET_IMAGES:
            cmd = input(f"[{count}/{TARGET_IMAGES}] > ")
            if cmd.strip().lower() == "q":
                break
            frame, when = coll.latest()
            if frame is None or when is None or time.monotonic() - when > 3.0:
                print("No fresh camera frame.")
                continue
            ok, corners, _ = _detect_corners(frame)
            if not ok:
                print("Checkerboard not detected.")
                continue
            blur = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                 cv2.CV_64F).var()
            if blur < BLUR_THRESHOLD:
                print(f"Too blurry (score {blur:.1f}).")
                continue

            i = _next_id()
            img_path = IMG_DIR / f"frame_{i:03d}.png"
            pre_path = PRE_DIR / f"corners_{i:03d}.png"
            cv2.imwrite(str(img_path), frame)
            prev = frame.copy()
            cv2.drawChessboardCorners(prev, CHECKERBOARD, corners, True)
            cv2.imwrite(str(pre_path), prev)
            count += 1
            print(f"Saved {img_path.name}  blur={blur:.1f}")
    finally:
        stop.set()
        coll.node.destroy_node()
        rclpy.shutdown()
    print(f"Done. {count} frames in {IMG_DIR}")


# -------- mode: solve --------

def solve():
    files = sorted(IMG_DIR.glob("frame_*.png"))
    if len(files) < MIN_IMAGES:
        print(f"Need at least {MIN_IMAGES} frames in {IMG_DIR}.")
        return

    template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32
    )
    template[:, :2] = (
        np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
        .T.reshape(-1, 2) * SQUARE_SIZE
    )

    obj_pts, img_pts, used = [], [], []
    size = None
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        cur = (img.shape[1], img.shape[0])
        if size is None:
            size = cur
        elif cur != size:
            print(f"skip {path.name}: size mismatch")
            continue
        ok, corners, _ = _detect_corners(img)
        if not ok:
            print(f"skip {path.name}: no corners")
            continue
        obj_pts.append(template.copy())
        img_pts.append(corners)
        used.append(path)

    if len(used) < MIN_IMAGES:
        print("Not enough valid frames.")
        return

    rms, K, D, rv, tv = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None
    )

    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rv[i], tv[i], K, D)
        d = img_pts[i].reshape(-1, 2) - proj.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))

    print(f"\nResolution: {size}")
    print(f"Valid frames: {len(used)}")
    print(f"RMS error: {rms:.4f} px")
    print(f"K =\n{K}")
    print(f"D = {D.ravel()}")

    info = {
        "image_width": size[0],
        "image_height": size[1],
        "camera_name": "robot_camera",
        "camera_matrix": {"rows": 3, "cols": 3, "data": K.ravel().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1, "cols": len(D.ravel()), "data": D.ravel().tolist()
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3, "data": np.eye(3).ravel().tolist()
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": np.hstack((K, np.zeros((3, 1)))).ravel().tolist()
        },
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"\nWrote {YAML_PATH}")


# -------- CLI --------

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("collect", "solve"):
        print("Usage: python -m robot.calibrate [collect|solve]")
        sys.exit(1)
    (collect if sys.argv[1] == "collect" else solve)()


if __name__ == "__main__":
    main()
