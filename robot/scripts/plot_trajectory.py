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
#   2. Neighbour bound (two-sided): a dot that is far from the median of
#      the NEIGHBOURS dots before and after it is noise. The allowed
#      distance is BASE_DEV plus MAX_SPEED times the time gap between
#      the dot and its neighbours, so sparse samples are not punished.
#      The filter repeats until no more dots are dropped.
#
# Usage:
#   python3 plot_trajectory.py [pose_csv] [png_path] [interval_s]
#                              [--show-outliers]

import sys
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

SHOW_OUTLIERS = "--show-outliers" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CSV = ARGS[0] if len(ARGS) > 0 else "/tmp/pose_log.csv"
PNG = ARGS[1] if len(ARGS) > 1 else "/tmp/trajectory_plot.png"
INTERVAL = float(ARGS[2]) if len(ARGS) > 2 else 0.5

ARENA_MARGIN = 0.60        # metres beyond waypoint/tag bounds
MAX_SPEED = 0.35           # m/s cap for a plausible step
BASE_DEV = 0.22            # metres allowed from the neighbour median
NEIGHBOURS = 3             # dots on each side used for the median

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

# Pass 2: neighbour clip. Compare each dot with the median of the dots
# before and after it. Unlike a check against only the previous dot, a
# single spike cannot pull the reference away, and a good dot after a
# spike is not dropped.


def neighbour_clip(pts):
    kept = []
    for i, (t, x, y) in enumerate(pts):
        lo = max(0, i - NEIGHBOURS)
        nb = [p for j, p in enumerate(pts[lo:i + NEIGHBOURS + 1], lo)
              if j != i]
        if len(nb) < 2:
            kept.append((t, x, y))
            continue
        tm = statistics.median(p[0] for p in nb)
        mx = statistics.median(p[1] for p in nb)
        my = statistics.median(p[2] for p in nb)
        dev = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
        if dev <= BASE_DEV + MAX_SPEED * abs(t - tm):
            kept.append((t, x, y))
    return kept


filtered = clean
while True:
    nxt = neighbour_clip(filtered)
    if len(nxt) == len(filtered):
        break
    filtered = nxt

kept_set = set(filtered)
outliers = [r for r in rows if r not in kept_set]

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

# Path line through the kept dots
ax.plot(xs, ys, color="steelblue", alpha=0.6,
        linewidth=1.5, label="path")

if SHOW_OUTLIERS and outliers:
    ax.scatter([r[1] for r in outliers], [r[2] for r in outliers],
               s=22, marker="x", color="grey", alpha=0.5,
               label=f"rejected ({len(outliers)})")

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
    f"{ts[-1] if ts else 0:.1f} s, {len(outliers)} outliers removed)"
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
