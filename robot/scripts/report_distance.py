#!/usr/bin/env python3
# Report how far the robot actually traveled and rotated.
#
# Two independent measures:
#
#   1. Pose log  — AprilTag estimates. Outlier filter removes jumps
#                  caused by ambiguity flips so the sum is not inflated.
#   2. Speed log — commanded wheel speeds, integrated over time.
#                  No pose-estimation noise, but biased by any slip or
#                  motor calibration error.
#
# The nominal route (0,0) -> (1,0) -> (1,2) -> (0,0) has a theoretical
# straight-line length of 1 + 2 + sqrt(5) = 5.236 m. The two in-route
# corners (90 deg and 135 deg) plus the final 135 deg heading change
# sum to 360 deg of nominal rotation.
#
# Report unit: 1 unit = 0.5 m (the minimum unit size allowed).
#
# Usage:
#   python3 report_distance.py [pose_csv] [speed_csv]

import sys
import csv
import math
from pathlib import Path

import yaml

POSE_CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pose_log.csv"
SPEED_CSV = sys.argv[2] if len(sys.argv) > 2 else "/tmp/speed_log.csv"
UNIT_M = 0.5                    # one report unit = 0.5 metres
SAMPLE_DT = 0.5                 # resample at this interval before summing
ARENA_MARGIN = 0.60
MAX_SPEED = 0.35
JUMP_MARGIN = 0.15

CONFIG_DIR = Path(
    "/home/ubuntu/ros2_ws/src/cse276a-robotics/robot/config"
)

# -------- motor model (must match config.py) --------
FORWARD_CMD = 0.20
FORWARD_SPEED = 0.195           # m/s at FORWARD_CMD
LEFT_CMD = 0.35
LEFT_90_S = 0.85                # 90 deg pivot takes this long at LEFT_CMD
RIGHT_CMD = 0.30
RIGHT_90_S = 1.00

# Speed per unit of command (metres per second per command unit).
SPEED_PER_CMD = FORWARD_SPEED / FORWARD_CMD

# Angular rate per unit of command difference (rad per second per cmd).
# A pivot runs the two wheels at +CMD and -CMD, so the differential is
# 2 * CMD. 90 deg in LEFT_90_S means rate = pi/2 / LEFT_90_S for a
# differential of 2 * LEFT_CMD.
RATE_PER_DIFF_LEFT = (math.pi / 2) / LEFT_90_S / (2 * LEFT_CMD)
RATE_PER_DIFF_RIGHT = (math.pi / 2) / RIGHT_90_S / (2 * RIGHT_CMD)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# -------- reference arena bounds (for pose outlier clip) --------
with open(CONFIG_DIR / "waypoints.yaml") as f:
    wp_cfg = yaml.safe_load(f)
scale = wp_cfg.get("scale", 1.0)
wps = [(x * scale, y * scale) for x, y, _ in wp_cfg["waypoints"]]

with open(CONFIG_DIR / "landmarks.yaml") as f:
    lm = yaml.safe_load(f)
tag_xy = [
    (t["position"][0], t["position"][1])
    for t in lm["tags"].values()
]

all_x = [p[0] for p in wps] + [p[0] for p in tag_xy]
all_y = [p[1] for p in wps] + [p[1] for p in tag_xy]
x_min, x_max = min(all_x) - ARENA_MARGIN, max(all_x) + ARENA_MARGIN
y_min, y_max = min(all_y) - ARENA_MARGIN, max(all_y) + ARENA_MARGIN


# -------- 1. nominal distance --------
nom_trans = 0.0
for a, b in zip(wps, wps[1:]):
    nom_trans += math.hypot(b[0] - a[0], b[1] - a[1])

nominal_corners_deg = 90.0 + 135.0 + 135.0


# -------- 2. actual translation and rotation from pose log --------
raw = []
with open(POSE_CSV) as f:
    for row in csv.DictReader(f):
        raw.append((
            float(row["t"]), float(row["x"]),
            float(row["y"]), float(row["theta"]),
        ))

filtered = []
for (t, x, y, th) in raw:
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        continue
    if not filtered:
        filtered.append((t, x, y, th))
        continue
    pt, px, py, _ = filtered[-1]
    dt = max(1e-3, t - pt)
    jump = math.hypot(x - px, y - py)
    if jump > MAX_SPEED * dt + JUMP_MARGIN:
        continue
    filtered.append((t, x, y, th))

# Resample at SAMPLE_DT so remaining jitter during a pause does not
# add up and inflate the sum. One sample per interval, first point kept.
resampled = []
next_t = 0.0
for (t, x, y, th) in filtered:
    if t >= next_t:
        resampled.append((t, x, y, th))
        next_t += SAMPLE_DT

pose_trans = 0.0
pose_rot = 0.0
for (a, b) in zip(resampled, resampled[1:]):
    pose_trans += math.hypot(b[1] - a[1], b[2] - a[2])
    pose_rot += abs(wrap(b[3] - a[3]))

pose_rot_deg = math.degrees(pose_rot)


# -------- 3. actual translation and rotation from speed log --------
cmd_trans = 0.0
cmd_rot = 0.0
prev_t = None
with open(SPEED_CSV) as f:
    for row in csv.DictReader(f):
        t = float(row["t"])
        left = float(row["left"])
        right = float(row["right"])
        if prev_t is not None:
            dt = t - prev_t
            if 0 < dt < 1.0:
                lin = ((left + right) / 2.0) * SPEED_PER_CMD
                cmd_trans += abs(lin) * dt
                diff = right - left
                rate_per_diff = (
                    RATE_PER_DIFF_LEFT if diff >= 0
                    else RATE_PER_DIFF_RIGHT
                )
                cmd_rot += abs(diff) * rate_per_diff * dt
        prev_t = t

cmd_rot_deg = math.degrees(cmd_rot)


# -------- print --------
def line(label, value, unit, in_units=True):
    extra = f"   ({value / UNIT_M:.2f} units)" if in_units else ""
    print(f"  {label:<32}{value:7.2f} {unit}{extra}")


print("=" * 64)
print(f"Distance report  (1 unit = {UNIT_M:.2f} m)")
print("=" * 64)
print()
print("Nominal (ideal straight-line path through all waypoints)")
line("translation",      nom_trans,           "m")
line("rotation (corners)", nominal_corners_deg, "deg", in_units=False)
print()
print(f"Actual, from AprilTag pose log "
      f"(outlier-filtered, resampled at {SAMPLE_DT:.1f} s)")
line("translation",      pose_trans,          "m")
line("rotation",         pose_rot_deg,        "deg", in_units=False)
print(f"  samples kept: {len(resampled)} "
      f"(from {len(filtered)} filtered, {len(raw)} raw)")
print()
print("Actual, from commanded wheel speeds (integrated)")
line("translation",      cmd_trans,           "m")
line("rotation",         cmd_rot_deg,         "deg", in_units=False)
print()
print("=" * 64)
