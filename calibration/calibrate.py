
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml
import rclpy

from sensor_msgs.msg import Image
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

# ============================================================
# CONFIGURATION
# ============================================================

CHECKERBOARD = (9, 6)      # Internal corners
SQUARE_SIZE = 0.024        # Meters (24 mm)

TARGET_IMAGES = 40
MIN_IMAGES = 10
BLUR_THRESHOLD = 35.0

CAMERA_TOPIC = "/camera/image_raw"

ROOT = Path("/home/ubuntu/ros2_ws/camera_calibration")
IMAGE_DIR = ROOT / "images"
PREVIEW_DIR = ROOT / "previews"
EXCLUDED_DIR = ROOT / "excluded_frames"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


# ============================================================
# CHECKERBOARD DETECTION
# ============================================================

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, flags
    )

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

    return found, corners, gray


def check_image(image):
    found, corners, gray = detect_corners(image)

    if not found:
        return False, None, "Checkerboard not detected"

    blur_score = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    if blur_score < BLUR_THRESHOLD:
        return (
            False,
            corners,
            f"Image too blurry (score: {blur_score:.1f})",
        )

    return True, corners, f"Blur score: {blur_score:.1f}"


# ============================================================
# ROS IMAGE CONVERSION
# ============================================================

def ros_to_opencv(msg):
    encoding = msg.encoding.lower()

    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding == "mono8":
        channels = 1
    else:
        raise ValueError(
            f"Unsupported image encoding: {msg.encoding}"
        )

    if msg.step < msg.width * channels:
        raise ValueError("Invalid image row stride")

    data = np.frombuffer(msg.data, dtype=np.uint8)

    if len(data) < msg.height * msg.step:
        raise ValueError("Incomplete ROS image buffer")

    if channels == 3:
        image = np.ndarray(
            (msg.height, msg.width, 3),
            dtype=np.uint8,
            buffer=data,
            strides=(msg.step, 3, 1),
        ).copy()

        if encoding == "rgb8":
            image = cv2.cvtColor(
                image, cv2.COLOR_RGB2BGR
            )
    else:
        image = np.ndarray(
            (msg.height, msg.width),
            dtype=np.uint8,
            buffer=data,
            strides=(msg.step, 1),
        ).copy()

        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGR
        )

    return image


# ============================================================
# SAFE FILE NUMBERING
# ============================================================

def get_existing_ids():
    """
    Include active, excluded, and preview images so that
    previously used frame IDs are never reused.
    """
    ids = set()

    directories = [
        (IMAGE_DIR, "frame_*.png"),
        (EXCLUDED_DIR, "frame_*.png"),
        (PREVIEW_DIR, "corners_*.png"),
    ]

    for directory, pattern in directories:
        for path in directory.glob(pattern):
            try:
                ids.add(int(path.stem.split("_")[-1]))
            except ValueError:
                pass

    return ids


def next_available_id():
    ids = get_existing_ids()
    return max(ids, default=0) + 1


# ============================================================
# MANUAL IMAGE COLLECTION
# ============================================================

class CameraCollector:

    def __init__(self):
        self.node = rclpy.create_node(
            "checkerboard_collector"
        )

        self.latest_frame = None
        self.latest_time = None
        self.lock = threading.Lock()

        self.subscription = self.node.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            CAMERA_QOS,
        )

    def image_callback(self, msg):
        try:
            frame = ros_to_opencv(msg)

            with self.lock:
                self.latest_frame = frame
                self.latest_time = time.monotonic()

        except Exception as error:
            print("Image conversion error:", error)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None

            return (
                self.latest_frame.copy(),
                self.latest_time,
            )

    def destroy(self):
        self.node.destroy_node()


def collect_images():
    rclpy.init()
    collector = CameraCollector()

    running = threading.Event()
    running.set()

    def spin_ros():
        while running.is_set() and rclpy.ok():
            rclpy.spin_once(
                collector.node,
                timeout_sec=0.1,
            )

    thread = threading.Thread(
        target=spin_ros,
        daemon=True,
    )
    thread.start()

    count = len(list(IMAGE_DIR.glob("frame_*.png")))

    print("\n======================================")
    print("CAMERA CALIBRATION - IMAGE COLLECTION")
    print("======================================")
    print(f"Directory: {ROOT}")
    print(f"Checkerboard: {CHECKERBOARD}")
    print(f"Square size: {SQUARE_SIZE * 1000:.0f} mm")
    print(f"Active images: {count}")
    print(f"Target active images: {TARGET_IMAGES}")
    print()
    print("Press ENTER to capture.")
    print("Type 'q' and press ENTER to quit.")
    print("Hold the board steady before each capture.")
    print("======================================\n")

    try:
        while count < TARGET_IMAGES:

            command = input(
                f"[{count}/{TARGET_IMAGES}] "
                "Press ENTER to capture (q to quit): "
            )

            if command.strip().lower() == "q":
                break

            frame, timestamp = collector.get_frame()

            if frame is None:
                print("No image received yet.\n")
                continue

            if (
                timestamp is None
                or time.monotonic() - timestamp > 3.0
            ):
                print("Camera frame is stale. Try again.\n")
                continue

            valid, corners, message = check_image(frame)

            if not valid:
                print(f"Rejected: {message}\n")
                continue

            index = next_available_id()

            image_path = (
                IMAGE_DIR / f"frame_{index:03d}.png"
            )

            preview_path = (
                PREVIEW_DIR / f"corners_{index:03d}.png"
            )

            preview = frame.copy()

            cv2.drawChessboardCorners(
                preview,
                CHECKERBOARD,
                corners,
                True,
            )

            # Exclusive creation prevents overwriting an
            # existing file, even if an ID is accidentally reused.
            success, encoded = cv2.imencode(".png", frame)

            if not success:
                print("Failed to encode image.\n")
                continue

            try:
                with open(image_path, "xb") as file:
                    file.write(encoded.tobytes())
            except FileExistsError:
                print("Filename already exists. Try again.\n")
                continue

            success, encoded_preview = cv2.imencode(
                ".png", preview
            )

            if success:
                try:
                    with open(preview_path, "xb") as file:
                        file.write(encoded_preview.tobytes())
                except FileExistsError:
                    print("Preview already exists; original saved.")

            count += 1

            print(f"Saved {count}/{TARGET_IMAGES}")
            print(f"Image: {image_path.name}")
            print(f"Preview: {preview_path.name}")
            print(f"{message}\n")

        print("\nCollection finished.")
        print(f"Active images: {count}")
        print(f"Image directory: {IMAGE_DIR}")
        print(f"Preview directory: {PREVIEW_DIR}")

    except (KeyboardInterrupt, EOFError):
        print("\nCollection interrupted.")

    finally:
        running.clear()
        thread.join(timeout=2.0)
        collector.destroy()
        rclpy.shutdown()


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def calibrate_camera():
    image_files = sorted(
        IMAGE_DIR.glob("frame_*.png")
    )

    if len(image_files) < MIN_IMAGES:
        print(
            f"Need at least {MIN_IMAGES} images. "
            f"Found {len(image_files)}."
        )
        return

    object_template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        dtype=np.float32,
    )

    object_template[:, :2] = (
        np.mgrid[
            0:CHECKERBOARD[0],
            0:CHECKERBOARD[1],
        ].T.reshape(-1, 2) * SQUARE_SIZE
    )

    object_points = []
    image_points = []
    accepted_files = []

    image_size = None

    print("\nDetecting checkerboard corners...\n")

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Cannot read {image_path.name}")
            continue

        current_size = (
            image.shape[1],
            image.shape[0],
        )

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Skipping {image_path.name}: "
                "resolution mismatch"
            )
            continue

        found, corners, _ = detect_corners(image)

        if not found:
            print(
                f"Skipping {image_path.name}: "
                "checkerboard not detected"
            )
            continue

        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted_files.append(image_path)

    if len(accepted_files) < MIN_IMAGES:
        print("Insufficient valid calibration images.")
        return

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    errors = []

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            K,
            D,
        )

        difference = (
            image_points[i].reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        error = np.sqrt(
            np.mean(
                np.sum(difference ** 2, axis=1)
            )
        )

        errors.append(error)

    print("\n======================================")
    print("CALIBRATION RESULTS")
    print("======================================")

    print(f"Resolution: {image_size}")
    print(f"Valid images: {len(accepted_files)}")
    print(f"RMS reprojection error: {rms:.4f} px")

    print("\nCamera matrix K:")
    print(K)

    print("\nDistortion coefficients D:")
    print(D.ravel())

    print("\nPer-image errors:")

    for path, error in zip(accepted_files, errors):
        print(f"{path.name}: {error:.4f} px")

    print(
        f"\nMean per-image error: "
        f"{np.mean(errors):.4f} px"
    )

    # ========================================================
    # SAVE CALIBRATION YAML
    # ========================================================

    R = np.eye(3)

    P = np.hstack(
        (K, np.zeros((3, 1)))
    )

    camera_info = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_name": "robot_camera",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": K.ravel().tolist(),
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(D.ravel()),
            "data": D.ravel().tolist(),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": R.ravel().tolist(),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": P.ravel().tolist(),
        },
    }

    yaml_path = ROOT / "camera_info.yaml"

    with open(yaml_path, "w") as file:
        yaml.safe_dump(
            camera_info,
            file,
            sort_keys=False,
        )

    # ========================================================
    # SAVE UNDISTORTION COMPARISON
    # ========================================================

    example = cv2.imread(
        str(accepted_files[0])
    )

    corrected = cv2.undistort(
        example, K, D
    )

    cv2.imwrite(
        str(ROOT / "original.png"),
        example,
    )

    cv2.imwrite(
        str(ROOT / "undistorted.png"),
        corrected,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = ROOT / "calibration_results.txt"

    with open(results_path, "w") as file:
        file.write(
            f"Resolution: {image_size}\n"
            f"Valid images: {len(accepted_files)}\n"
            f"RMS error: {rms:.6f} px\n\n"
            f"Camera matrix:\n{K}\n\n"
            f"Distortion coefficients:\n{D.ravel()}\n\n"
        )

        for path, error in zip(
            accepted_files, errors
        ):
            file.write(
                f"{path.name}: {error:.6f} px\n"
            )

    print("\nSaved:")
    print(yaml_path)
    print(results_path)
    print(ROOT / "original.png")
    print(ROOT / "undistorted.png")
    print("\nCalibration complete.")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python3 calibrate.py "
            "[collect|calibrate]"
        )
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "collect":
        collect_images()
    elif mode == "calibrate":
        calibrate_camera()
    else:
        print("Invalid mode. Use collect or calibrate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
