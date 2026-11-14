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
# straight-line length of 1 + 2 + sqrt(5) = 5.236 m. The planned turns
# are computed the same way navigator.py computes them: 90 deg at WP1,
# 153.4 deg at WP2 and a 116.6 deg final align, 360 deg in total.
#
# The speed log is also split into separate pivot turns (the two wheels
# turn in opposite directions). Each turn is classed as a planned turn
# (waypoint turn or final align) or a heading correction.
#
# Report unit: 1 unit = 0.5 m (the minimum unit size allowed).
#
# Usage:
#   python3 report_distance.py [pose_csv] [speed_csv]
#   python3 report_distance.py none [speed_csv]   # wheel data only

import sys
import csv
import math
import statistics
from pathlib import Path

import yaml

POSE_CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pose_log.csv"
SPEED_CSV = sys.argv[2] if len(sys.argv) > 2 else "/tmp/speed_log.csv"
UNIT_M = 0.5                    # one report unit = 0.5 metres
SAMPLE_DT = 0.5                 # resample at this interval before summing
ARENA_MARGIN = 0.60
MAX_SPEED = 0.35
BASE_DEV = 0.22                 # same filter as plot_trajectory.py
NEIGHBOURS = 3

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

# Planned turns, same rule as navigator._handle_reach / _align_final.
wps_full = [(x * scale, y * scale, th) for x, y, th in wp_cfg["waypoints"]]
planned_turns = []
for i in range(1, len(wps_full) - 1):
    (px, py, _), (ax_, ay_, _), (nx, ny, _) = wps_full[i - 1:i + 2]
    planned_turns.append(math.degrees(wrap(
        math.atan2(ny - ay_, nx - ax_) - math.atan2(ay_ - py, ax_ - px))))
(px, py, _), (fx, fy, fth) = wps_full[-2], wps_full[-1]
planned_turns.append(math.degrees(wrap(fth - math.atan2(fy - py, fx - px))))
planned_turns = [a for a in planned_turns if abs(a) > 15.0]
nominal_corners_deg = sum(abs(a) for a in planned_turns)


# -------- 2. actual translation and rotation from pose log --------
pose_trans = pose_rot_deg = None
USE_POSE = POSE_CSV.lower() != "none" and Path(POSE_CSV).exists()
if USE_POSE:
    raw = []
    with open(POSE_CSV) as f:
        for row in csv.DictReader(f):
            raw.append((
                float(row["t"]), float(row["x"]),
                float(row["y"]), float(row["theta"]),
            ))

    # Same two-sided neighbour filter as plot_trajectory.py, so the
    # numbers agree with the plot.
    def neighbour_clip(pts):
        kept = []
        for i, p in enumerate(pts):
            lo = max(0, i - NEIGHBOURS)
            nb = [q for j, q in enumerate(pts[lo:i + NEIGHBOURS + 1], lo)
                  if j != i]
            if len(nb) < 2:
                kept.append(p)
                continue
            tm = statistics.median(q[0] for q in nb)
            mx = statistics.median(q[1] for q in nb)
            my = statistics.median(q[2] for q in nb)
            dev = math.hypot(p[1] - mx, p[2] - my)
            if dev <= BASE_DEV + MAX_SPEED * abs(p[0] - tm):
                kept.append(p)
        return kept


    filtered = [r for r in raw
                if x_min <= r[1] <= x_max and y_min <= r[2] <= y_max]
    while True:
        nxt = neighbour_clip(filtered)
        if len(nxt) == len(filtered):
            break
        filtered = nxt

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


# -------- 4. split the speed log into separate pivot turns --------
DEG_PER_DIFF_LEFT = math.degrees(RATE_PER_DIFF_LEFT)
DEG_PER_DIFF_RIGHT = math.degrees(RATE_PER_DIFF_RIGHT)
MAX_CORRECTION_DEG = 30.0       # navigator caps a correction at this

spd = []
with open(SPEED_CSV) as f:
    for row in csv.DictReader(f):
        spd.append((float(row["t"]), float(row["left"]),
                    float(row["right"])))

turns = []                      # dicts: start, end, deg (+ = left)
for i, (t, left, right) in enumerate(spd):
    if left * right >= 0:
        continue                # not a pivot
    dt = spd[i + 1][0] - t if i + 1 < len(spd) else 0.05
    dt = min(dt, 0.2)
    diff = right - left
    deg = diff * (DEG_PER_DIFF_LEFT if diff > 0 else DEG_PER_DIFF_RIGHT) * dt
    last = turns[-1] if turns else None
    if last and t - last["end"] < 0.15 and (deg > 0) == (last["deg"] > 0):
        last["deg"] += deg
        last["end"] = t + dt
    else:
        turns.append({"start": t, "end": t + dt, "deg": deg})

# Class each turn. A planned turn is a pivot bigger than the correction
# cap. navigator.split_long_turn splits a turn over 90 deg into 90 + rest,
# so the pivot right after a 90 deg pivot (same direction) is its tail.
for k, tr in enumerate(turns):
    prev = turns[k - 1] if k > 0 else None
    is_tail = (prev is not None and abs(prev["deg"]) >= 85.0
               and (prev["deg"] > 0) == (tr["deg"] > 0)
               and tr["start"] - prev["end"] < 0.6)
    if abs(tr["deg"]) > MAX_CORRECTION_DEG + 5.0:
        tr["kind"] = "planned"
    elif is_tail:
        tr["kind"] = "planned"
    else:
        tr["kind"] = "correction"

main_rot_deg = sum(abs(t["deg"]) for t in turns if t["kind"] == "planned")
corr_rot_deg = sum(abs(t["deg"]) for t in turns if t["kind"] == "correction")
n_corr = sum(1 for t in turns if t["kind"] == "correction")
net_rot_deg = sum(t["deg"] for t in turns)


# -------- 5. total moving distance INCLUDING rotation --------
# Count the distance the wheels travel: (|left| + |right|) / 2 speed.
# Straight driving gives the forward distance. A pivot on the spot gives
# the arc each wheel travels, which is angle x HALF_TRACK.
# HALF_TRACK (wheel to robot centre) comes from the same calibration:
# wheel speed at the pivot command / pivot turn rate.
HALF_TRACK = ((LEFT_CMD * SPEED_PER_CMD) / ((math.pi / 2) / LEFT_90_S)
              + (RIGHT_CMD * SPEED_PER_CMD) / ((math.pi / 2) / RIGHT_90_S)
              ) / 2.0

wheel_travel = 0.0
for (t, left, right), (t2, _, _) in zip(spd, spd[1:]):
    dt = t2 - t
    if 0 < dt < 1.0:
        wheel_travel += (abs(left) + abs(right)) / 2.0 * SPEED_PER_CMD * dt
turn_travel = wheel_travel - cmd_trans          # metres spent turning
# Split the measured turning distance by angle share, so parts add up.
_all = main_rot_deg + corr_rot_deg
turn_travel_main = turn_travel * (main_rot_deg / _all if _all else 0.0)
turn_travel_corr = turn_travel - turn_travel_main
nom_turn_travel = math.radians(nominal_corners_deg) * HALF_TRACK
nom_total = nom_trans + nom_turn_travel


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
line("rotation (planned turns)", nominal_corners_deg, "deg",
     in_units=False)
print("  planned turns: " + ", ".join(f"{a:+.1f}" for a in planned_turns)
      + " deg")
print()
if USE_POSE:
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
print("Every pivot turn the robot made (from wheel commands)")
print(f"  {'#':>2}  {'time (s)':>13}  {'dir':<5} {'angle':>7}  kind")
for k, tr in enumerate(turns, 1):
    print(f"  {k:>2}  {tr['start']:5.1f} - {tr['end']:5.1f}  "
          f"{'LEFT' if tr['deg'] > 0 else 'RIGHT':<5} "
          f"{abs(tr['deg']):6.1f}°  {tr['kind']}")
line("planned turns", main_rot_deg, "deg", in_units=False)
line(f"heading corrections ({n_corr})", corr_rot_deg, "deg", in_units=False)
line("ALL turns (total rotation)", main_rot_deg + corr_rot_deg, "deg",
     in_units=False)
line("net heading change (L - R)", net_rot_deg, "deg", in_units=False)
print()
print(f"TOTAL MOVING DISTANCE, rotation included "
      f"(half-track {HALF_TRACK:.3f} m from calibration)")
line("planned: driving", nom_trans, "m")
line("planned: turning", nom_turn_travel, "m")
line("planned: TOTAL", nom_total, "m")
line("actual: driving", cmd_trans, "m")
line("actual: turning", turn_travel, "m")
line("   - planned turns", turn_travel_main, "m")
line(f"   - corrections ({n_corr})", turn_travel_corr, "m")
line("actual: TOTAL", wheel_travel, "m")
print()
print("=" * 64)
