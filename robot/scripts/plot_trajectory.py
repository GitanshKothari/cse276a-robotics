#!/usr/bin/env python3
# Plot the robot 2D trajectory from the algorithm pose log.
#
# The pose log contains every AprilTag estimate the navigator accepted.
# Some of those estimates are ambiguity flips that are physically
# impossible (the robot cannot teleport across the arena). We drop them
# with two sanity rules before plotting:
#
#   1. Arena bound: a dot that falls far outside the waypoint and tag
#      bounding box is noise.
#   2. Motion bound: a dot that moves faster than MAX_SPEED metres per
#      second from the previous accepted dot is noise.
#
# One dot per sample interval (default 0.5 s).
#
# Usage:
#   python3 plot_trajectory.py [pose_csv] [png_path] [interval_s]

import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pose_log.csv"
PNG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/trajectory_plot.png"
INTERVAL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

ARENA_MARGIN = 0.60        # metres beyond waypoint/tag bounds
MAX_SPEED = 0.35           # m/s cap for a plausible step
JUMP_MARGIN = 0.15         # metres of grace on top of MAX_SPEED * dt

CONFIG_DIR = Path(
    "/home/ubuntu/ros2_ws/src/cse276a-robotics/robot/config"
)

# ---- reference waypoints and tags ----
with open(CONFIG_DIR / "waypoints.yaml") as f:
    wp_cfg = yaml.safe_load(f)
scale = wp_cfg.get("scale", 1.0)
wps = [(x * scale, y * scale) for x, y, _ in wp_cfg["waypoints"]]

with open(CONFIG_DIR / "landmarks.yaml") as f:
    lm = yaml.safe_load(f)
tags = {
    int(tid): (t["position"][0], t["position"][1])
    for tid, t in lm["tags"].items()
}

# ---- arena bounds for sanity clip ----
all_x = [p[0] for p in wps] + [p[0] for p in tags.values()]
all_y = [p[1] for p in wps] + [p[1] for p in tags.values()]
x_min, x_max = min(all_x) - ARENA_MARGIN, max(all_x) + ARENA_MARGIN
y_min, y_max = min(all_y) - ARENA_MARGIN, max(all_y) + ARENA_MARGIN


def in_arena(x, y):
    return x_min <= x <= x_max and y_min <= y <= y_max


# ---- read algorithm pose log ----
rows = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        rows.append((float(row["t"]), float(row["x"]), float(row["y"])))

# ---- drop impossible samples ----
# Pass 1: arena clip.
clean = [(t, x, y) for (t, x, y) in rows if in_arena(x, y)]

# Pass 2: motion clip. A dot that is too far from the previous accepted
# dot (based on elapsed time) is treated as an ambiguity flip.
filtered = []
for (t, x, y) in clean:
    if not filtered:
        filtered.append((t, x, y))
        continue
    pt, px, py = filtered[-1]
    dt = max(1e-3, t - pt)
    jump = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
    if jump > MAX_SPEED * dt + JUMP_MARGIN:
        continue
    filtered.append((t, x, y))

ts = [r[0] for r in filtered]
xs = [r[1] for r in filtered]
ys = [r[2] for r in filtered]

# ---- sample one dot every INTERVAL seconds ----
sx, sy = [], []
next_t = 0.0
for t, x, y in zip(ts, xs, ys):
    if t >= next_t:
        sx.append(x)
        sy.append(y)
        next_t += INTERVAL

# ---- draw ----
fig, ax = plt.subplots(figsize=(8, 8))

# Full trajectory (filtered) as a faint line
ax.plot(xs, ys, color="steelblue", alpha=0.4,
        linewidth=1.0, label="path")

# Sampled dots
ax.scatter(sx, sy, s=28, color="steelblue",
           zorder=3, label=f"pose every {INTERVAL:.1f} s")

# Waypoints
wx = [p[0] for p in wps]
wy = [p[1] for p in wps]
ax.plot(wx, wy, "x", color="red", markersize=12,
        markeredgewidth=2.0, label="waypoints")
for i, (x, y) in enumerate(wps):
    ax.annotate(f"WP{i}", (x, y), xytext=(6, 6),
                textcoords="offset points", color="red")

# Tags
if tags:
    tx = [p[0] for p in tags.values()]
    ty = [p[1] for p in tags.values()]
    ax.plot(tx, ty, "s", color="goldenrod", markersize=8,
            label="AprilTags")
    for tid, (x, y) in tags.items():
        ax.annotate(f"tag {tid}", (x, y), xytext=(6, -10),
                    textcoords="offset points", color="goldenrod")

# Start and end markers
if sx:
    ax.plot(sx[0], sy[0], "o", color="green",
            markersize=12, label="start")
    ax.plot(sx[-1], sy[-1], "o", color="black",
            markersize=12, label="end")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(
    f"Robot 2D trajectory ({len(sx)} samples, "
    f"{ts[-1] if ts else 0:.1f} s)"
)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(PNG, dpi=120)

dropped = len(rows) - len(filtered)
print(
    f"Saved {PNG} ({len(sx)} sampled dots, "
    f"{len(filtered)} kept, {dropped} dropped as outliers)"
)
