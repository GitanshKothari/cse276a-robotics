"""The known world poses of the AprilTag landmarks."""

import math

import yaml

from .transforms import tag_extrinsics


class Landmark:
    def __init__(self, tag_id, pose, size):
        self.tag_id = tag_id
        self.pose = pose      # 4x4 world <- tag
        self.size = size      # edge length of the black square, metres

    @property
    def position(self):
        return self.pose[:3, 3]


class LandmarkMap:
    """Tag id -> world pose, read from the map yaml.

    Each entry gives the centre of the tag and ``normal_yaw``, the compass
    direction in the world xy plane that the printed face looks towards. A
    tag on the far wall of the arena facing back at the origin therefore
    has ``normal_yaw: 180``.
    """

    def __init__(self, landmarks, default_size, cans=None):
        self.landmarks = landmarks
        self.default_size = default_size
        # Coke cans, id -> (x, y). A can is a cylinder standing on the
        # floor, so it has no orientation and no height to record: the
        # filter only ever uses its ground position.
        self.cans = cans or {}

    @classmethod
    def from_file(cls, path):
        with open(path) as handle:
            document = yaml.safe_load(handle) or {}
        return cls.from_dict(document)

    @classmethod
    def from_dict(cls, document):
        default_size = float(document.get('tag_size', 0.165))
        landmarks = {}

        for entry in document.get('landmarks') or []:
            tag_id = int(entry['id'])
            size = float(entry.get('size', default_size))
            pose = tag_extrinsics(
                float(entry['x']),
                float(entry['y']),
                float(entry.get('z', 0.0)),
                math.radians(float(entry.get('normal_yaw', 0.0))),
                math.radians(float(entry.get('tilt', 0.0))),
            )
            landmarks[tag_id] = Landmark(tag_id, pose, size)

        cans = {
            int(entry['id']): (float(entry['x']), float(entry['y']))
            for entry in document.get('cans') or []
        }
        return cls(landmarks, default_size, cans)

    def __contains__(self, tag_id):
        return int(tag_id) in self.landmarks

    def __len__(self):
        return len(self.landmarks)

    def get(self, tag_id):
        return self.landmarks.get(int(tag_id))

    def size_of(self, tag_id):
        landmark = self.get(tag_id)
        return landmark.size if landmark else self.default_size

    def ids(self):
        return sorted(self.landmarks)

    def can_ids(self):
        return sorted(self.cans)
