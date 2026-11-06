"""Task decisions from RGB-D measurements only, with explicit unknown outcomes."""
import numpy as np


class PickupVerifier:
    """Require measured lift and a stable cube-to-tool offset across fresh frames.

    Thresholds are initial engineering choices, not measured calibration accuracy.
    Raw RGB-D positions are used so filter lag cannot manufacture rigid motion.
    """

    def __init__(self, initial_cube, min_lift=0.035, offset_tolerance=0.025,
                 required_frames=4):
        self.initial = np.asarray(initial_cube).copy()
        self.min_lift = min_lift
        self.tolerance = offset_tolerance
        self.required = required_frames
        self.offsets = []
        self.heights = []

    def observe(self, cube_position, tcp_position, tcp_rotation, gripper_holding):
        if cube_position is None:
            self.offsets.clear()
            self.heights.clear()
            return "unconfirmed"
        cube = np.asarray(cube_position)
        if not gripper_holding:
            self.offsets.clear()
            self.heights.clear()
            return "failed"
        if cube[2] - self.initial[2] < self.min_lift:
            self.offsets.clear()
            self.heights.clear()
            return "unconfirmed"
        offset = np.asarray(tcp_rotation).T @ (cube - tcp_position)
        if np.linalg.norm(offset) > 0.10:
            self.offsets.clear()
            return "failed"
        self.offsets.append(offset)
        self.heights.append(float(cube[2]))
        self.offsets = self.offsets[-self.required:]
        if len(self.offsets) < self.required:
            return "unconfirmed"
        offsets = np.asarray(self.offsets)
        if np.max(np.linalg.norm(offsets - np.mean(offsets, axis=0), axis=1)) > self.tolerance:
            return "unconfirmed"
        return "confirmed"


def placement_observation(cube_position, tray_position, gripper_open):
    """Conservative centre-in-interior test; unavailable vision stays unknown."""
    if cube_position is None or tray_position is None:
        return "unconfirmed"
    cube, tray = np.asarray(cube_position), np.asarray(tray_position)
    inside = np.all(np.abs(cube[:2] - tray[:2]) <= 0.027)
    settled_height = 0.008 <= cube[2] <= 0.055
    return "confirmed" if inside and settled_height and gripper_open else "failed"
