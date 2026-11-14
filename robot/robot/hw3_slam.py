"""Planar EKF-SLAM with camera position as the robot state."""

import math

import numpy as np


J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rotation(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]])


def measurement_R(forward, left):
    distance = max(0.20, abs(forward))
    angle_factor = (abs(left) / distance) ** 2

    sigma_forward = 0.05 * (1.0 + 0.20 * distance) * (1.0 + angle_factor)
    sigma_left = 0.05 * (1.0 + 0.15 * distance) * (1.0 + 1.5 * angle_factor)
    return np.diag([sigma_forward ** 2, sigma_left ** 2])


class Slam:
    def __init__(self):
        self.x = np.zeros(3)  # Camera x, y, heading; then landmark x, y pairs
        self.P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(3) ** 2])
        self.index = {}
        self.camera_offset = np.array([0.11, 0.0])

    def predict(self, distance, turn):
        heading = self.x[2]
        new_heading = heading + turn
        R0 = rotation(heading)
        R1 = rotation(new_heading)
        forward = R0[:, 0]
        offset = self.camera_offset

        # The wheels rotate about the axle, but the state tracks the camera.
        camera_move = distance * forward + (R1 - R0) @ offset

        n = len(self.x)
        F = np.eye(n)
        F[:2, 2] = distance * (J @ forward) + (J @ R1 - J @ R0) @ offset

        G = np.zeros((n, 2))
        G[:2, 0] = forward
        G[:2, 1] = J @ R1 @ offset
        G[2, 1] = 1.0

        turning = abs(turn) >= abs(distance)
        distance_noise = 0.04 if turning else 0.08
        turn_noise = 0.15 if turning else 0.05
        base_noise = 0.0 if turning else 0.015

        Q = np.diag([
            (base_noise + distance_noise * abs(distance) + 0.01 * abs(turn)) ** 2,
            (base_noise + distance_noise * abs(distance) + turn_noise * abs(turn)) ** 2,
        ])

        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self.x[:2] += camera_move
        self.x[2] = new_heading

    def observe(self, tag, forward, left, update_landmark=True):
        z = np.array([forward, left])
        rot = rotation(self.x[2])
        R = measurement_R(forward, left)

        if tag not in self.index:
            A = np.zeros((2, len(self.x)))
            A[:, :2] = np.eye(2)
            A[:, 2] = rot @ J @ z

            cross = A @ self.P
            self.index[tag] = len(self.x)
            self.x = np.r_[self.x, self.x[:2] + rot @ z]
            self.P = np.block([
                [self.P, cross.T],
                [cross, cross @ A.T + rot @ R @ rot.T],
            ])
            return

        i = self.index[tag]
        expected = rot.T @ (self.x[i:i + 2] - self.x[:2])

        H = np.zeros((2, len(self.x)))
        H[:, :2] = -rot.T
        H[:, 2] = -J @ expected
        H[:, i:i + 2] = rot.T

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        if not update_landmark:
            K[i:i + 2] = 0.0

        self.x += K @ (z - expected)
        IKH = np.eye(len(self.x)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def landmarks(self):
        result = {}
        for tag, i in self.index.items():
            result[tag] = {
                "xy": self.x[i:i + 2].tolist(),
                "cov": self.P[i:i + 2, i:i + 2].tolist(),
            }
        return result
