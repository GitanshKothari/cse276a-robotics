"""Plot and score HW3 laps in the measured world frame.

Usage (single):
  hw3_evaluate.py lap.json truth.csv --start X Y DEG

Usage (side-by-side report figure):
  hw3_evaluate.py square.json octagon.json truth.csv --start X Y DEG \\
      --out combined.png --captions "(a) Square" "(b) Octagon"
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np


def required_path(sides, side_m):
    x = y = 0.0
    pts = [(0.0, 0.0)]
    for k in range(sides):
        h = k * 2 * math.pi / sides
        x += side_m * math.cos(h)
        y += side_m * math.sin(h)
        pts.append((x, y))
    pts.append((0.0, 0.0))
    return np.array(pts)


def load_truth(path):
    with open(path) as f:
        return {r["id"]: np.array([float(r["x"]), float(r["y"])])
                for r in csv.DictReader(f)}


def style_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.9, zorder=0)
    ax.grid(True, which="minor", color="#f3f3f3", linewidth=0.5, zorder=0)
    ax.minorticks_on()
    ax.tick_params(axis="both", labelsize=10, length=4, width=0.8, colors="#333333")
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(1.05)
    ax.set_xlabel("World $x$ (m)", fontsize=12)
    ax.set_ylabel("World $y$ (m)", fontsize=12)


def draw_panel(ax, data, truth, start, rot, caption):
    shape = "Square" if data["sides"] == 4 else "Octagon"
    path_local = np.array(data["path"])[:, :2]
    path = path_local @ rot.T + start

    if data.get("path_reference") == "axle":
        offset = np.asarray(
            data.get("camera_offset", [0.11, 0.0]),
            dtype=float
        )

        axle_start_world = start - rot @ offset

        req = (
            required_path(data["sides"], data["side_m"])
            @ rot.T + axle_start_world
        )
    else:
        req = (
            required_path(data["sides"], data["side_m"])
            @ rot.T + start
        )

    ax.plot(*req.T, linestyle=(0, (6, 3)), color="#0b6e4f", linewidth=2.6,
            solid_capstyle="round", zorder=3,
            label=f"Required path ({data['side_m']:.2f} m sides)")
    ax.plot(*path.T, "-", color="#1d3557", linewidth=2.0, zorder=4,
            label="Estimated robot path")
    ax.scatter(path[1:, 0], path[1:, 1], s=16, color="#1d3557",
               edgecolors="white", linewidths=0.4, zorder=5)
    ax.scatter([path[0, 0]], [path[0, 1]], s=64, marker="o",
               color="#1d3557", edgecolors="white", linewidths=1.2, zorder=6)

    errors = []
    for i, (tag, gpos) in enumerate(truth.items()):
        ax.plot(gpos[0], gpos[1], "x", color="#111111", markersize=8.5,
                markeredgewidth=1.9, zorder=7,
                label="Ground-truth landmark" if i == 0 else None)
        ax.annotate(str(tag), gpos, xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color="#111111", fontweight="medium", zorder=8)

    for i, (tag, landmark) in enumerate(data["tags"].items()):
        pos = rot @ np.asarray(landmark["xy"]) + start
        cov = rot @ np.asarray(landmark["cov"]) @ rot.T
        values, vectors = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(5.991 * np.maximum(values, 0))
        tilt = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        ax.add_patch(Ellipse(
            pos, width, height, angle=tilt, fill=True,
            facecolor="#e63946", edgecolor="none", alpha=0.10, zorder=2))
        ax.add_patch(Ellipse(
            pos, width, height, angle=tilt, fill=False,
            edgecolor="#c1121f", linewidth=1.15, alpha=0.85, zorder=3))
        ax.plot(pos[0], pos[1], "o", color="#c1121f", markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=7,
                label="Estimated landmark" if i == 0 else None)
        ax.annotate(str(tag), pos, xytext=(5, -12), textcoords="offset points",
                    fontsize=9, color="#9b2226", zorder=8)
        if tag in truth:
            err = float(np.linalg.norm(pos - truth[tag]))
            errors.append(err)
            print(f"  Tag {tag}: {err:.3f} m")

    mean = float(np.mean(errors)) if errors else float("nan")
    print(f"  Matched {len(errors)}/{len(truth)}  mean={mean:.3f} m")
    print("  Missed:", sorted(truth.keys() - data["tags"].keys()))

    # Diagnostic only: error of the map shape after a rigid best fit
    # (rotation + shift). A large gap to the mean above = start pose error.
    common = [t for t in data["tags"] if t in truth]
    if len(common) >= 3:
        a = np.array([data["tags"][t]["xy"] for t in common])
        b = np.array([truth[t] for t in common])
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        if np.linalg.det(vt.T @ u.T) < 0:
            vt[-1] *= -1
        fit_rot = vt.T @ u.T
        shift = cb - fit_rot @ ca
        fit = np.linalg.norm(a @ fit_rot.T + shift - b, axis=1)
        fit_deg = np.degrees(np.arctan2(fit_rot[1, 0], fit_rot[0, 0]))
        print(f"  Map shape (best rigid fit): mean={fit.mean():.3f} m "
              f"max={fit.max():.3f} m; fitted start = "
              f"({shift[0]:.2f}, {shift[1]:.2f}, {fit_deg:+.1f} deg)")

    style_axes(ax)
    pts = [req, path] + [truth[t] for t in truth]
    pts += [rot @ np.asarray(lm["xy"]) + start for lm in data["tags"].values()]
    pts = np.vstack(pts)
    pad = 0.28
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    mean_cm = mean * 100.0 if errors else float("nan")
    ax.set_title(
        f"{caption}\n{shape}  |  side = {data['side_m']:.2f} m  |  "
        f"mean landmark error = {mean_cm:.1f} cm",
        fontsize=11.5, pad=10, loc="left")
    return mean


def legend_handles():
    return [
        Line2D([0], [0], color="#0b6e4f", lw=2.6, linestyle=(0, (6, 3)),
               label="Required path"),
        Line2D([0], [0], color="#1d3557", lw=2.0, label="Estimated robot path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d3557",
               markeredgecolor="white", markersize=8, label="Start pose"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="#111111",
               markeredgewidth=2, markersize=8, label="Ground-truth landmark"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c1121f",
               markeredgecolor="white", markersize=7, label="Estimated landmark"),
        Patch(facecolor="#e63946", edgecolor="#c1121f", alpha=0.25,
              label="95% landmark covariance"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="one or more lap JSON files, then the ground-truth CSV last")
    parser.add_argument("--start", type=float, nargs=3, required=True,
                        metavar=("X_M", "Y_M", "HEADING_DEG"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path")
    parser.add_argument("--captions", nargs="*", default=None,
                        help="panel captions, e.g. '(a) Square trajectory' '(b) Octagon trajectory'")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("provide lap JSON file(s) and a ground-truth CSV as the last argument")

    truth_path = args.inputs[-1]
    lap_paths = args.inputs[:-1]
    if truth_path.suffix.lower() != ".csv":
        parser.error("last argument must be the ground-truth CSV")
    if not lap_paths:
        parser.error("provide at least one lap JSON before the CSV")

    truth = load_truth(truth_path)
    start = np.array(args.start[:2])
    angle = np.radians(args.start[2])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    laps = [json.loads(p.read_text()) for p in lap_paths]
    n = len(laps)
    letters = "abcdefghijklmnopqrstuvwxyz"
    if args.captions and len(args.captions) == n:
        captions = list(args.captions)
    else:
        captions = []
        for i, data in enumerate(laps):
            shape = "Square" if data["sides"] == 4 else "Octagon"
            captions.append(f"({letters[i]}) {shape} trajectory")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.6, 7.4), dpi=160)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n, figsize=(6.8 * n + 0.6, 7.0), dpi=160)
        axes = list(np.atleast_1d(axes))

    means = []
    for ax, data, lap_path, cap in zip(axes, laps, lap_paths, captions):
        print(f"{lap_path}:")
        means.append(draw_panel(ax, data, truth, start, rot, cap))

    fig.legend(handles=legend_handles(), loc="lower center", ncol=3,
               frameon=True, fancybox=False, edgecolor="#cccccc",
               fontsize=10, framealpha=1.0, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "CSE 276 HW3 — EKF-SLAM: required trajectory vs estimated map",
        fontsize=15, fontweight="semibold", y=0.98)

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    if args.out is not None:
        out = args.out
    elif n == 1:
        out = lap_paths[0].with_suffix(".png")
    else:
        out = Path.home() / "hw3" / "hw3_report_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    if means:
        print("Panel means (m):", [round(m, 3) for m in means])


if __name__ == "__main__":
    main()
