"""Draw the 2D trajectory figure the report asks for.

One dot per logged sample, so with the logger's default half-second
period the spacing between dots reads directly as speed. Everything
plotted is the filter's estimate, not a measured-by-eye path.
"""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the validated default palette. A scatter uses
# the all-pairs pairlist, where only the first three slots clear the
# colour-vision floors, so the figure is capped at three series.
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dedcd6'
PATH_COLOR = '#2a78d6'
WAYPOINT_COLOR = '#eb6834'
LANDMARK_COLOR = '#1baf7a'


def read_trajectory(path):
    samples = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            samples.append((
                float(row['t']), float(row['x']), float(row['y']), float(row['theta']),
                int(row['tags_in_view']) if row.get('tags_in_view') else 0,
            ))
    return samples


def read_waypoints(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def read_landmarks(path):
    if not path or not os.path.exists(path):
        return []
    import yaml
    with open(path) as handle:
        document = yaml.safe_load(handle) or {}
    return [(int(e['id']), float(e['x']), float(e['y']))
            for e in document.get('landmarks') or []]


def plot(run_dir, map_path, targets, output):
    samples = read_trajectory(os.path.join(run_dir, 'trajectory.csv'))
    if not samples:
        raise SystemExit(f'No samples in {run_dir}/trajectory.csv')
    reached = read_waypoints(os.path.join(run_dir, 'waypoint_errors.csv'))
    landmarks = read_landmarks(map_path)

    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    blind = [s[4] == 0 for s in samples]

    figure, axes = plt.subplots(figsize=(7.2, 6.4), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(GRID)

    # Faint thread so the order of the dots is unambiguous; the dots
    # themselves carry the timing.
    axes.plot(xs, ys, color=PATH_COLOR, linewidth=2.0, alpha=0.35, zorder=2)

    # Open dots mark the samples taken with no tag in view, which is where
    # the estimate is running on dead reckoning alone.
    axes.scatter([x for x, b in zip(xs, blind) if not b],
                 [y for y, b in zip(ys, blind) if not b],
                 s=46, color=PATH_COLOR, edgecolors=SURFACE, linewidths=1.6,
                 zorder=4, label='Estimated pose (every 0.5 s)')
    if any(blind):
        axes.scatter([x for x, b in zip(xs, blind) if b],
                     [y for y, b in zip(ys, blind) if b],
                     s=46, facecolors=SURFACE, edgecolors=PATH_COLOR,
                     linewidths=1.6, zorder=4, label='No landmark in view')

    if targets:
        axes.scatter([t[0] for t in targets], [t[1] for t in targets],
                     s=190, marker='X', color=WAYPOINT_COLOR,
                     edgecolors=SURFACE, linewidths=1.8, zorder=5,
                     label='Commanded waypoint')
        for index, (tx, ty, ttheta) in enumerate(targets):
            axes.annotate(f'W{index}', (tx, ty), textcoords='offset points',
                          xytext=(10, 8), fontsize=10, color=TEXT_PRIMARY,
                          fontweight='bold')
            axes.arrow(tx, ty, 0.18 * math.cos(ttheta), 0.18 * math.sin(ttheta),
                       head_width=0.05, head_length=0.05, fc=WAYPOINT_COLOR,
                       ec=WAYPOINT_COLOR, alpha=0.75, zorder=5,
                       length_includes_head=True)

    if landmarks:
        axes.scatter([m[1] for m in landmarks], [m[2] for m in landmarks],
                     s=150, marker='s', color=LANDMARK_COLOR,
                     edgecolors=SURFACE, linewidths=1.8, zorder=5,
                     label='AprilTag landmark')
        for tag_id, mx, my in landmarks:
            axes.annotate(f'tag {tag_id}', (mx, my), textcoords='offset points',
                          xytext=(9, -14), fontsize=9, color=TEXT_SECONDARY)

    # Where the rover thought it was when it declared each waypoint reached.
    for row in reached:
        ex, ey = float(row['estimate_x']), float(row['estimate_y'])
        tx, ty = float(row['target_x']), float(row['target_y'])
        axes.plot([tx, ex], [ty, ey], color=TEXT_SECONDARY, linewidth=1.2,
                  linestyle=':', zorder=3)

    axes.set_xlabel('x (m)', color=TEXT_SECONDARY)
    axes.set_ylabel('y (m)', color=TEXT_SECONDARY)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axes.set_title('Estimated trajectory', color=TEXT_PRIMARY,
                   fontsize=13, fontweight='bold', loc='left', pad=12)

    duration = samples[-1][0]
    axes.set_title(f'{duration:.0f} s run, dot every 0.5 s',
                   color=TEXT_SECONDARY, fontsize=10, loc='right', pad=12)

    # Equal aspect, or the path's shape is a lie.
    axes.set_aspect('equal', adjustable='datalim')
    legend = axes.legend(loc='best', frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    figure.tight_layout()
    figure.savefig(output, facecolor=SURFACE)
    print(f'Wrote {output}')

    summary_path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            summary = json.load(handle)
        print(json.dumps(summary, indent=2))

    if reached:
        print('\nwaypoint   target                 estimate               '
              'pos err (m)  yaw err (deg)')
        for row in reached:
            print('  %-8s (%5.2f,%5.2f,%5.2f)  (%5.2f,%5.2f,%5.2f)  %10s  %12s' % (
                row['index'], float(row['target_x']), float(row['target_y']),
                float(row['target_theta']), float(row['estimate_x']),
                float(row['estimate_y']), float(row['estimate_theta']),
                row['position_error'], row['heading_error_deg']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='directory written by trajectory_logger')
    parser.add_argument('--map', dest='map_path', default='',
                        help='landmark yaml, to draw the tags')
    parser.add_argument('--waypoints', default='0,0,0 1,0,0 1,2,3.14159 0,0,0',
                        help='space separated x,y,theta triples')
    parser.add_argument('-o', '--output', default='')
    args = parser.parse_args(argv)

    targets = []
    for chunk in args.waypoints.split():
        parts = chunk.split(',')
        if len(parts) == 3:
            targets.append(tuple(float(p) for p in parts))

    output = args.output or os.path.join(args.run_dir, 'trajectory.png')
    plot(args.run_dir, args.map_path, targets, output)


if __name__ == '__main__':
    main()
