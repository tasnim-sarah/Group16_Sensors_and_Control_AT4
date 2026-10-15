# Reusable wrist RGB-D sensing for Group 16's three-cube MuJoCo sorting scene.

# No ground-truth cube/body poses are used for detection. The model/data/renderer
# belong to the caller; this module neither steps nor resets the simulation.
# 

import numpy as np
import cv2
import mujoco

COLOUR_RANGES = {
    "Red Cube": [([0, 130, 100], [10, 255, 255]), ([170, 130, 100], [179, 255, 255])],
    "Yellow Cube": [([22, 180, 150], [38, 255, 255])],
    "Blue Cube": [([100, 170, 90], [130, 255, 255])],
}

# Pickup workspace in WORLD metres, based on the cube placement bands.
# This is a task-space constraint, not MuJoCo ground-truth object lookup.
DEFAULT_PICKUP_BOUNDS = (0.24, 0.46, -0.23, 0.23)  # xmin, xmax, ymin, ymax


def add_wrist_camera(spec):
   
    wrist = spec.body("wrist_3_link")
    if wrist is None:
        raise ValueError("wrist_3_link missing from the scene")
    # Intended for the supplied scene, which has no wrist_cam in XML.
    cam = wrist.add_camera(name="wrist_cam", pos=[0.0325, 0.095, 0.05], fovy=51.28)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]
    return cam


def aim_camera_once(model, data, target_world=(0.35, -0.05, 0.03), camera_name="wrist_cam"):
    # """Set a FIXED wrist-to-camera mounting orientation at current observation pose.

    # Must be performed only when configuring the camera, not each frame: after
    # this the camera moves with the wrist (it does NOT track a target).
    # """
    mujoco.mj_forward(model, data)
    cam_id = model.camera(camera_name).id
    wrist_id = model.body("wrist_3_link").id
    cam_pos = data.cam_xpos[cam_id].copy()
    forward = np.asarray(target_world, dtype=float) - cam_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-10:
        raise ValueError("Camera and target positions coincide")
    forward /= norm
    z_axis = -forward  # MuJoCo camera looks along its local -Z axis
    reference = np.array([0., 0., 1.])
    if abs(float(np.dot(reference, z_axis))) > 0.98:
        reference = np.array([0., 1., 0.])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_wc = np.column_stack((x_axis, y_axis, z_axis))
    R_ww = data.xmat[wrist_id].reshape(3, 3)
    R_wrist_cam = R_ww.T @ R_wc
    mujoco.mju_mat2Quat(model.cam_quat[cam_id], R_wrist_cam.ravel())
    mujoco.mj_forward(model, data)
    alignment = np.dot(-data.cam_xmat[cam_id].reshape(3, 3)[:, 2], forward)
    return float(alignment)


class CameraSensor:
    def __init__(self, model, renderer, camera_name="wrist_cam",
                 pickup_bounds=DEFAULT_PICKUP_BOUNDS, min_area=80):
        self.model = model
        self.renderer = renderer
        self.camera_name = camera_name
        self.cam_id = model.camera(camera_name).id
        self.height = int(renderer.height)
        self.width = int(renderer.width)
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        self.fy = self.height / (2.0 * np.tan(fovy / 2.0))
        self.fx = self.fy  # square pixels in MuJoCo's pinhole renderer
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0
        self.pickup_bounds = pickup_bounds
        self.min_area = min_area
        self.kernel = np.ones((3, 3), dtype=np.uint8)

    def capture(self, data):
      
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        return rgb, depth

    def detect(self, data, rgb=None, depth=None):
        
        if rgb is None or depth is None:
            rgb, depth = self.capture(data)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        cam_pos = data.cam_xpos[self.cam_id].copy()
        R_wc = data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        xmin, xmax, ymin, ymax = self.pickup_bounds
        detections = {}
        for label, ranges in COLOUR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                                    np.array(hi, dtype=np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (self.min_area <= area <= self.width * self.height * 0.10):
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                u = moments["m10"] / moments["m00"]
                v = moments["m01"] / moments["m00"]
                selected = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(selected, [contour], -1, 255, -1)
                interior = cv2.erode(selected, self.kernel, iterations=2)
                if not interior.any():
                    interior = selected
                depths = depth[interior > 0]
                valid = depths[np.isfinite(depths) & (depths > 0)]
                if valid.size < 10:
                    continue
                d = float(np.median(valid))
                p_cam = np.array([(u - self.cx) * d / self.fx,
                                  -(v - self.cy) * d / self.fy, -d])
                p_world = R_wc @ p_cam + cam_pos
                # The colour-coded bins are OUTSIDE the pickup area.
                if not (xmin <= p_world[0] <= xmax and ymin <= p_world[1] <= ymax):
                    continue
                candidates.append((area, {
                    "pixel": (float(u), float(v)),
                    "depth_m": d,
                    "position_world_m": p_world,
                    "area_px": float(area),
                    "valid_depth_pixels": int(valid.size),
                    "contour": contour,
                }))
            if candidates:
                detections[label] = max(candidates, key=lambda item: item[0])[1]
        return detections


def selected_target(detections, colour):
    return detections.get(f"{colour.strip().capitalize()} Cube")
