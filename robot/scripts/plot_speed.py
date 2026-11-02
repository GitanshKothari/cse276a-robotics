#!/usr/bin/env python3
# Plot the wheel-command speed-time curve written by navigator.py.
# Usage: python3 plot_speed.py [csv_path] [png_path]

import sys
import csv
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/speed_log.csv"
PNG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/speed_plot.png"

ts, L, R = [], [], []
with open(CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts.append(float(row["t"]))
        L.append(float(row["left"]))
        R.append(float(row["right"]))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts, L, label="left wheel cmd", linewidth=1.3)
ax.plot(ts, R, label="right wheel cmd", linewidth=1.3)
ax.set_xlabel("time (s)")
ax.set_ylabel("wheel command")
ax.set_title("Robot wheel command vs time")
ax.axhline(0.0, color="grey", linewidth=0.5)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PNG, dpi=120)
print(f"Saved {PNG} ({len(ts)} samples)")
