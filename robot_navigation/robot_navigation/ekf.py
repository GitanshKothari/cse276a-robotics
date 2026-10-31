"""Extended Kalman filter over the planar pose (x, y, theta).

Prediction runs off the twist the controller commanded, because the rover
has no wheel encoders. That makes the motion model the weak part of the
estimate, so its noise is modelled as proportional to how hard the rover
is being driven, and the AprilTag updates are what keep it honest.
"""

import math

import numpy as np

from .transforms import wrap_angle


class PoseEKF:
    def __init__(self, initial_pose=(0.0, 0.0, 0.0), initial_std=(0.02, 0.02, 0.02),
                 alpha_v=0.25, beta_v=0.02, alpha_w=0.25, beta_w=0.05):
        self.state = np.array(initial_pose, dtype=float)
        self.covariance = np.diag(np.square(np.asarray(initial_std, dtype=float)))

        # Control noise grows with the commanded speed (alpha) on top of a
        # floor that covers wheel slip and the uneven floor (beta).
        self.alpha_v = alpha_v
        self.beta_v = beta_v
        self.alpha_w = alpha_w
        self.beta_w = beta_w

    @property
    def pose(self):
        return float(self.state[0]), float(self.state[1]), float(self.state[2])

    @property
    def position_std(self):
        return math.sqrt(max(self.covariance[0, 0] + self.covariance[1, 1], 0.0))

    @property
    def heading_std(self):
        return math.sqrt(max(self.covariance[2, 2], 0.0))

    def predict(self, v, omega, dt):
        """Propagate one step of the unicycle model."""
        if dt <= 0.0:
            return

        theta = self.state[2]
        # Integrate about the midpoint heading, which is a much better
        # approximation than the start heading when the rover is turning.
        mid = theta + 0.5 * omega * dt
        cos_mid, sin_mid = math.cos(mid), math.sin(mid)

        self.state[0] += v * dt * cos_mid
        self.state[1] += v * dt * sin_mid
        self.state[2] = wrap_angle(theta + omega * dt)

        # Jacobian of the motion model with respect to the state.
        jacobian = np.eye(3)
        jacobian[0, 2] = -v * dt * sin_mid
        jacobian[1, 2] = v * dt * cos_mid

        # Jacobian with respect to the control, so control noise can be
        # pushed through into state noise.
        control_jacobian = np.array([
            [dt * cos_mid, -0.5 * v * dt * dt * sin_mid],
            [dt * sin_mid, 0.5 * v * dt * dt * cos_mid],
            [0.0, dt],
        ])
        control_noise = np.diag([
            (self.alpha_v * abs(v) + self.beta_v) ** 2,
            (self.alpha_w * abs(omega) + self.beta_w) ** 2,
        ])

        self.covariance = (
            jacobian @ self.covariance @ jacobian.T
            + control_jacobian @ control_noise @ control_jacobian.T
        )

    def update_pose(self, measurement, noise, gate=None):
        """Fuse a full (x, y, theta) observation.

        Returns the squared Mahalanobis distance, or None when the
        measurement was rejected by the gate.
        """
        observation = np.eye(3)
        innovation = np.asarray(measurement, dtype=float) - self.state
        innovation[2] = wrap_angle(innovation[2])
        return self._correct(innovation, observation, noise, gate)

    def update_range_bearing(self, landmark_xy, measurement, noise, gate=None):
        """Fuse the range and bearing of one landmark of known position.

        This is the strong half of an AprilTag sighting. Range and bearing
        alone do not pin the heading down from a single tag, but two tags
        in the same frame do, and the filter gets there on its own by
        applying them one after the other.
        """
        terms = self._range_bearing_terms(landmark_xy, measurement)
        if terms is None:
            return None
        innovation, observation = terms
        return self._correct(innovation, observation, noise, gate)

    def _range_bearing_terms(self, landmark_xy, measurement):
        """Innovation and observation matrix for one sighting, or None."""
        dx = float(landmark_xy[0]) - self.state[0]
        dy = float(landmark_xy[1]) - self.state[1]
        squared = dx * dx + dy * dy
        if squared < 1e-9:
            return None
        expected_range = math.sqrt(squared)

        innovation = np.array([
            float(measurement[0]) - expected_range,
            wrap_angle(float(measurement[1]) - wrap_angle(math.atan2(dy, dx) - self.state[2])),
        ])
        observation = np.array([
            [-dx / expected_range, -dy / expected_range, 0.0],
            [dy / squared, -dx / squared, -1.0],
        ])
        return innovation, observation

    def range_bearing_mahalanobis(self, landmark_xy, measurement, noise):
        """Score a sighting against a landmark without changing the state.

        An AprilTag carries its own id, so the filter never has to ask
        which landmark it is looking at. A coke can carries nothing: every
        can looks the same. So the filter has to choose, and it chooses by
        scoring each candidate first. A wrong match is worse than no match,
        because it pulls the estimate towards a landmark that is not there
        and makes the next match wronger still.
        """
        terms = self._range_bearing_terms(landmark_xy, measurement)
        if terms is None:
            return None
        innovation, observation = terms
        innovation_cov = observation @ self.covariance @ observation.T + noise
        try:
            inverse = np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            return None
        return float(innovation @ inverse @ innovation)

    def update_heading(self, heading, variance, gate=None):
        """Fuse a heading-only observation, e.g. the IMU yaw."""
        observation = np.array([[0.0, 0.0, 1.0]])
        innovation = np.array([wrap_angle(heading - self.state[2])])
        return self._correct(innovation, observation, np.array([[variance]]), gate)

    def _correct(self, innovation, observation, noise, gate):
        innovation_cov = observation @ self.covariance @ observation.T + noise
        try:
            inverse = np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            return None

        distance = float(innovation @ inverse @ innovation)
        if gate is not None and distance > gate:
            return None

        gain = self.covariance @ observation.T @ inverse
        self.state = self.state + gain @ innovation
        self.state[2] = wrap_angle(self.state[2])

        # Joseph form, which stays symmetric and positive definite even
        # after many updates with a nearly singular innovation covariance.
        identity_minus = np.eye(3) - gain @ observation
        self.covariance = (
            identity_minus @ self.covariance @ identity_minus.T
            + gain @ noise @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return distance

    def inflate(self, position_std, heading_std):
        """Admit the estimate is worse than the covariance claims.

        A filter that is lost but confident rejects every measurement that
        would save it: the gate is computed from the covariance, and a
        small covariance makes a large innovation look impossible. Nothing
        is then accepted, so nothing shrinks the error, and the filter
        stays lost. Widening the covariance reopens the gate.

        This matters far more for cans than for tags. A tag announces its
        id, so a tag sighting is still usable when the pose is well out.
        A can sighting has to be matched first, and matching is what fails.
        """
        self.covariance[0, 0] += position_std ** 2
        self.covariance[1, 1] += position_std ** 2
        self.covariance[2, 2] += heading_std ** 2

    def reset(self, pose, std=(0.02, 0.02, 0.02)):
        self.state = np.array(pose, dtype=float)
        self.covariance = np.diag(np.square(np.asarray(std, dtype=float)))
