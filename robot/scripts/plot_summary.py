#!/usr/bin/env python3
# One-page summary chart: planned vs actual moving distance and rotation.
#
# Uses ONLY the wheel command log (speed_log.csv). The AprilTag pose log
# is not used. The numbers come from report_distance.py, so the chart
# and the text report always agree.
#
# Left panel : total moving distance in units, rotation included
#              (driving + distance the wheels travel while turning).
# Right panel: total rotation in degrees.
# Two panels, because units and degrees cannot share an axis.
#
# Usage:
#   python3 plot_summary.py [speed_csv] [png_path]

import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt

SPEED_CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/speed_log.csv"
PNG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/summary_plot.png"

# ---- get the numbers from report_distance.py (wheel data only) ----
sys.argv = ["report_distance.py", "none", SPEED_CSV]
with redirect_stdout(io.StringIO()):
    r = runpy.run_path(str(Path(__file__).with_name("report_distance.py")))
U = r["UNIT_M"]

# ---- palette (reference light palette) ----
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
PLAN_DRIVE = "#9a9993"   # neutral greys: the plan is a reference
PLAN_TURN = "#cfcec8"
DRIVE = "#2a78d6"        # series 1: driving straight
TURN = "#eb6834"         # series 2: planned turns
CORR = "#f5b393"         # lighter step of series 2: corrections

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT,
    "hatch.color": SURFACE,
})

fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(13, 4.2),
                                 facecolor=SURFACE)


def stack(ax, y, parts):
    """parts: list of (value, color, hatch, legend_label)."""
    left = 0.0
    for v, c, h, lab in parts:
        ax.barh(y, v, left=left, height=0.45, color=c, hatch=h,
                edgecolor=SURFACE, linewidth=2, zorder=2, label=lab)
        left += v
    return left


def style(ax, title, unit, labels, vmax):
    ax.set_facecolor(SURFACE)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, 2.0)
    ax.set_xlim(0, vmax * 1.5)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                 pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.5, 1.0))


def label(ax, x, y, text, bold=False):
    ax.text(x, y, text, va="center", fontsize=10.5, color=TEXT,
            fontweight="bold" if bold else "normal")


# ---- left: total moving distance, rotation included (units) ----
plan_drive = r["nom_trans"] / U
plan_turn = r["nom_turn_travel"] / U
act_drive = r["cmd_trans"] / U
act_main = r["turn_travel_main"] / U
act_corr = r["turn_travel_corr"] / U

plan_tot = stack(ax_d, 1, [
    (plan_drive, PLAN_DRIVE, None, None),
    (plan_turn, PLAN_TURN, None, None),
])
act_tot = stack(ax_d, 0, [
    (act_drive, DRIVE, None, f"driving {act_drive:.2f}"),
    (act_main, TURN, None, f"planned turns {act_main:.2f}"),
    (act_corr, CORR, "///", f"{r['n_corr']} corrections {act_corr:.2f}"),
])
vmax = max(plan_tot, act_tot)
pad = vmax * 0.02
label(ax_d, plan_tot + pad, 1,
      f"{plan_tot:.2f} units  (driving {plan_drive:.2f} + turns "
      f"{plan_turn:.2f})")
label(ax_d, act_tot + pad, 0,
      f"{act_tot:.2f} units  ({act_tot / plan_tot * 100:.0f}% of plan)",
      bold=True)
style(ax_d, "Total moving distance (rotation included)",
      f"units  (1 unit = {U:.1f} m)", ["Planned", "Robot (actual)"], vmax)

# ---- right: total rotation (degrees) ----
plan_r = r["nominal_corners_deg"]
main, corr = r["main_rot_deg"], r["corr_rot_deg"]
stack(ax_r, 1, [(plan_r, PLAN_TURN, None, None)])
total = stack(ax_r, 0, [
    (main, TURN, None, f"3 planned turns {main:.0f}°"),
    (corr, CORR, "///", f"{r['n_corr']} corrections {corr:.0f}°"),
])
vmax = max(plan_r, total)
pad = vmax * 0.02
label(ax_r, plan_r + pad, 1, f"{plan_r:.0f}°")
label(ax_r, total + pad, 0,
      f"{total:.0f}°  ({total / plan_r * 100:.0f}% of plan)", bold=True)
style(ax_r, "Total rotation", "degrees",
      ["Planned", "Robot (actual)"], vmax)

fig.suptitle(
    "Robot run: planned vs actual  (route (0,0) → (1,0) → (1,2) → (0,0))",
    x=0.01, ha="left", fontsize=14, fontweight="bold",
)
fig.text(
    0.01, 0.015,
    "Data: wheel command log only (wheel speed × time, motor calibration "
    "from config.py). Turning distance = angle × "
    f"{r['HALF_TRACK']:.3f} m (wheel-to-centre distance from the same "
    "calibration).\nThe robot made the 3 planned turns plus small heading "
    "corrections (max 30° each); these corrections are the extra distance.",
    ha="left", va="bottom", fontsize=9.5, color=TEXT_2,
)
fig.tight_layout(rect=(0, 0.11, 1, 0.94), w_pad=4)
fig.savefig(PNG, dpi=150, facecolor=SURFACE)
print(f"Saved {PNG}")
