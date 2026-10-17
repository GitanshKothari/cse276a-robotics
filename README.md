## Summary
Adds an open-loop trajectory follower for `waypoints.txt`. The planner turns each leg into stop, spin-in-place, and straight commands. The controller only prints that list and publishes it on `motor_commands`. It does not use the camera or the IMU.

A turn is kept between −180° and 180°, so the rover takes the short way around. A turn over 90° is split once: the return toward home is 90° left, then 45° left, then two straight drives. Left and right use different wheel commands because the rover does not spin at the same rate in both directions. Those times live in `config.py`.

Wheel commands stay inside the motor limit of ±0.50. Left is −0.40 / +0.40. Right is +0.45 / −0.45. Straight is 0.30 on both wheels.

## Plan for the checked-in waypoints
Running the planner on `waypoints.txt` prints this schedule. Total time is 32.01 s.

| t (s) | duration | move | L | R |
| --- | --- | --- | --- | --- |
| 0.00 | 3.40 | straight | +0.30 | +0.30 |
| 3.40 | 0.50 | stop | 0 | 0 |
| 3.90 | 1.19 | left 90° | −0.40 | +0.40 |
| 5.09 | 3.40 | straight | +0.30 | +0.30 |
| 8.49 | 0.50 | stop | 0 | 0 |
| 8.99 | 1.62 | right 90° | +0.45 | −0.45 |
| 10.61 | 3.40 | straight | +0.30 | +0.30 |
| 14.01 | 0.50 | stop | 0 | 0 |
| 14.51 | 1.19 | left 90° | −0.40 | +0.40 |
| 15.70 | 3.40 | straight | +0.30 | +0.30 |
| 19.11 | 0.50 | stop | 0 | 0 |
| 19.61 | 1.19 | left 90° | −0.40 | +0.40 |
| 20.80 | 0.50 | stop | 0 | 0 |
| 21.30 | 0.59 | left 45° | −0.40 | +0.40 |
| 21.89 | 4.81 | straight | +0.30 | +0.30 |
| 26.70 | 4.81 | straight | +0.30 | +0.30 |
| 31.51 | 0.50 | stop | 0 | 0 |

Only the heading on the first waypoint is used. The two home legs stay separate. Both are forward commands, so the rover does not stop between them.

## How the times were chosen
Open-loop spins were timed on the rover, then scaled so a measured angle maps back to 90°.

- Right: a 90° hold was short, about 80°. `RIGHT_90_S` is `1.44 × 90/80 = 1.62` s.
- Left: a 1.78 s hold on the first left reached about 135°. `LEFT_90_S` is `1.78 × 90/135 = 1.19` s. The 45° piece is half of that, 0.59 s.
- Straight: `FORWARD_SPEED` is 0.294 m/s at command 0.30, so 1 m takes 3.40 s and each home leg takes 4.81 s.

The 1.19 s left time has not been driven again since that change. On the previous run, at 1.78 s, the first left was about 135° and the last 90° left was a little under 90°. One shared left time cannot correct both.

## Test
Motor node already running:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
python3 -m robot_control.trajectory_controller
```

Check that the printed plan matches the table above, then that the wheels stop at the end. Ctrl-C also sends three zero commands before exit.

# Testing Evidence:
Youtube Link: https://youtu.be/JqYXS2vwTBs
