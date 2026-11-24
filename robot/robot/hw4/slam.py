import math
import numpy as np
import arena

J = np.array([[0., -1.], [1., 0.]])

def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])

def wrap(angle):
    return math.remainder(angle, 2 * math.pi)


class Slam:
    def __init__(self):
        self.x = np.array(arena.START, dtype=float)
        self.P = np.diag([0.025**2, 0.025**2, 0.08**2])
        self.indices = {}
        self.views = {}
        self.offset = np.array(arena.CAMERA_OFFSET)

    def predict(self, distance=0.0, turn=0.0):
        forward = rotation(self.x[2])[:, 0]
        delta = distance * forward
        F = np.eye(len(self.x))
        F[:2, 2] = J @ delta
        self.P = F @ self.P @ F.T
        self.P[:2, :2] += np.outer(forward, forward) * (0.002 + 0.16 * abs(distance))**2
        self.P[2, 2] += (0.005 + 0.12 * abs(turn) + 0.03 * abs(distance))**2
        self.x[:2] += delta
        self.x[2] = wrap(self.x[2] + turn)

    def observe(self, tag, z):
        body = rotation(self.x[2])
        camera = rotation(self.x[2] + arena.CAMERA_YAW)
        camera_xy = self.x[:2] + body @ self.offset
        noise = np.eye(2) * (0.035 * (1 + 0.20 * (z @ z)))**2
        index = self.indices.get(tag)

        if tag not in arena.BOUNDARY_TAGS and index is None:
            A = np.zeros((2, len(self.x)))
            A[:, :2] = np.eye(2)
            A[:, 2] = body @ J @ self.offset + camera @ J @ z
            cross = A @ self.P
            
            self.indices[tag] = len(self.x)
            self.x = np.r_[self.x, camera_xy + camera @ z]
            self.P = np.block([[self.P, cross.T], [cross, A @ cross.T + camera @ noise @ camera.T]])
            self.views[tag] = [self.x[:2].copy()]
            return True

        xy = np.array(arena.BOUNDARY_TAGS[tag]) if index is None else self.x[index:index + 2]
        expected = camera.T @ (xy - camera_xy)
        
        H = np.zeros((2, len(self.x)))
        H[:, :2] = -camera.T
        H[:, 2] = -J @ expected - camera.T @ body @ J @ self.offset
        
        if index is not None:
            H[:, index:index + 2] = camera.T
        innovation = z - expected
        S = H @ self.P @ H.T + noise
        
        if innovation @ np.linalg.solve(S, innovation) > 9.21:
            return False
        
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        self.x[2] = wrap(self.x[2])
        
        A = np.eye(len(self.x)) - K @ H
        self.P = A @ self.P @ A.T + K @ noise @ K.T
        self.P = (self.P + self.P.T) / 2
        
        if index is not None:
            self.views[tag].append(self.x[:2].copy())
        return True

    def position_sigma(self):
        return math.sqrt(max(0., np.linalg.eigvalsh(self.P[:2, :2])[-1]))

    def cylinder(self):
        tags = [tag for tag, views in self.views.items() if len(views) >= 2]
        
        if len(tags) < 3:
            raise RuntimeError("Need three cylinder tags, each used at least twice.")
        
        indices = [i for tag in tags for i in (self.indices[tag], self.indices[tag] + 1)]
        landmarks = self.x[indices].reshape(-1, 2)
        relative = landmarks - landmarks.mean(axis=0)
        design = np.column_stack([2 * relative, np.ones(len(tags))])
        
        if np.linalg.matrix_rank(design) < 3:
            raise RuntimeError("Cylinder tag centres are collinear.")
        
        fit = np.linalg.lstsq(design, np.sum(relative**2, axis=1), rcond=None)[0]
        center = landmarks.mean(axis=0) + fit[:2]
        distances = np.linalg.norm(landmarks - center, axis=1)
        
        tag_radius = float(distances.mean())
        radius = tag_radius - arena.TAG_STANDOFF
        residual = float(max(abs(distances - tag_radius)))
        
        angles = np.sort(np.arctan2(landmarks[:, 1] - center[1], landmarks[:, 0] - center[0]))
        coverage = 2 * math.pi - max(np.diff(np.r_[angles, angles[0] + 2 * math.pi]))
        
        if coverage < math.radians(120) or not 0.05 < radius < 0.75 or residual > 0.06:
            raise RuntimeError("Cylinder fit failed coverage, radius, or residual check.")
        unit = (landmarks - center) / distances[:, None]
        fit_jacobian = np.linalg.pinv(np.column_stack((unit, np.ones(len(tags)))))
        radial_jacobian = np.zeros((len(tags), 2 * len(tags)))
        
        for i, direction in enumerate(unit):
            radial_jacobian[i, 2*i:2*i+2] = direction
        
        H = fit_jacobian @ radial_jacobian
        covariance = H @ self.P[np.ix_(indices, indices)] @ H.T
        center_sigma = math.sqrt(max(0., np.linalg.eigvalsh(covariance[:2, :2])[-1]))
        
        radius_sigma = math.sqrt(max(0., covariance[2, 2]))
        padding = 3 * (center_sigma + radius_sigma) + residual + 0.015
        
        return {"center": center.tolist(), "radius": radius, "padding": padding,
                "residual": residual, "tags": dict(zip(tags, landmarks.tolist()))}
