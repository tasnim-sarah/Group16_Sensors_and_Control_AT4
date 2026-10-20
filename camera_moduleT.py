import numpy as np
import cv2
import mujoco


# HSV colour ranges for cube detection
COLOUR_RANGES = {
    "Red Cube": [
        ([0, 130, 100], [10, 255, 255]),
        ([170, 130, 100], [179, 255, 255]),
    ],
    "Yellow Cube": [
        ([22, 180, 150], [38, 255, 255]),
    ],
    "Blue Cube": [
        ([100, 170, 90], [130, 255, 255]),
    ],
}

# Pickup workspace in world metres: xmin, xmax, ymin, ymax
DEFAULT_PICKUP_BOUNDS = (0.24, 0.46, -0.23, 0.23)


# --------------------------------------------------
# 1. POSITION TRACKER (constant-position Kalman filter + gating)
# --------------------------------------------------

class PositionTracker:
    """3-state Kalman filter for a cube that sits still on the table.

    State: x, y, z (metres). The model is "the cube stays where it is", with
    a small process noise that lets it drift if it gets nudged.

    Measurements are gated: if a new measurement is further than `gate` metres
    from the estimate, the cube is assumed to have MOVED and the filter is
    re-initialised on the new measurement instead of lagging behind it.

    The covariance is a single scalar (same variance on x, y and z).
    """

    def __init__(self, meas_std=0.003, process_std=0.0005, gate=0.02,
                 init_std=0.05):
        self.r = float(meas_std) ** 2        # default measurement variance
        self.q = float(process_std) ** 2     # process variance per update
        self.gate = float(gate)
        self.init_var = float(init_std) ** 2
        self.x = None
        self.p = self.init_var
        self.updates = 0
        self.resets = 0

    def reset(self):
        self.x = None
        self.p = self.init_var
        self.updates = 0

    def update(self, z, meas_var=None):
        z = np.asarray(z, dtype=float).ravel()
        r = self.r if meas_var is None else float(meas_var)

        if self.x is None:
            self.x = z.copy()
            self.p = r
            self.updates = 1
            return self.x.copy()

        innovation = z - self.x
        if np.linalg.norm(innovation) > self.gate:
            # Cube moved (or a bad earlier estimate): restart from this value.
            self.resets += 1
            self.x = z.copy()
            self.p = r
            self.updates = 1
            return self.x.copy()

        p_pred = self.p + self.q
        k = p_pred / (p_pred + r)
        self.x = self.x + k * innovation
        self.p = (1.0 - k) * p_pred
        self.updates += 1
        return self.x.copy()

    @property
    def std(self):
        """1-sigma uncertainty of the estimate in metres."""
        return float(np.sqrt(self.p))


# --------------------------------------------------
# 2. WRIST CAMERA SETUP
# --------------------------------------------------

def add_wrist_camera(spec):
    wrist = spec.body("wrist_3_link")

    if wrist is None:
        raise ValueError("wrist_3_link missing from the scene")

    cam = wrist.add_camera(
        name="wrist_cam",
        pos=[0.0325, 0.095, 0.05],
        fovy=51.28
    )

    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]

    return cam


def aim_camera_once(
    model,
    data,
    target_world=(0.35, -0.05, 0.03),
    camera_name="wrist_cam"
):
    """Aim the camera once when configuring the scene."""

    mujoco.mj_forward(model, data)

    cam_id = model.camera(camera_name).id
    wrist_id = model.body("wrist_3_link").id

    cam_pos = data.cam_xpos[cam_id].copy()

    forward = np.asarray(target_world, dtype=float) - cam_pos
    norm = np.linalg.norm(forward)

    if norm < 1e-10:
        raise ValueError("Camera and target positions coincide")

    forward /= norm

    z_axis = -forward

    reference = np.array([0., 0., 1.])

    if abs(float(np.dot(reference, z_axis))) > 0.98:
        reference = np.array([0., 1., 0.])

    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    R_wc = np.column_stack((x_axis, y_axis, z_axis))
    R_ww = data.xmat[wrist_id].reshape(3, 3)

    R_wrist_cam = R_ww.T @ R_wc

    mujoco.mju_mat2Quat(
        model.cam_quat[cam_id],
        R_wrist_cam.ravel()
    )

    mujoco.mj_forward(model, data)

    alignment = np.dot(
        -data.cam_xmat[cam_id].reshape(3, 3)[:, 2],
        forward
    )

    return float(alignment)


# --------------------------------------------------
# 3. CAMERA SENSOR
# --------------------------------------------------

class CameraSensor:

    def __init__(
        self,
        model,
        renderer,
        camera_name="wrist_cam",
        pickup_bounds=DEFAULT_PICKUP_BOUNDS,
        min_area=80,
        depth_noise_std=0.0,    # metres, per depth pixel (0 = ideal sensor)
        pixel_noise_std=0.0,    # pixels, on the detected centroid
        meas_std=0.003,         # assumed 1-sigma of one raw XYZ measurement (m)
        gate=0.02,              # innovation gate for the tracker (m)
        seed=None,
    ):
        self.model = model
        self.renderer = renderer
        self.camera_name = camera_name

        self.cam_id = model.camera(camera_name).id

        self.height = int(renderer.height)
        self.width = int(renderer.width)

        fovy = np.deg2rad(model.cam_fovy[self.cam_id])

        self.fy = self.height / (2.0 * np.tan(fovy / 2.0))
        self.fx = self.fy

        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

        self.pickup_bounds = pickup_bounds
        self.min_area = min_area

        self.depth_noise_std = float(depth_noise_std)
        self.pixel_noise_std = float(pixel_noise_std)
        self.rng = np.random.default_rng(seed)

        self.kernel = np.ones((3, 3), dtype=np.uint8)

        # One tracker per colour
        self.trackers = {
            label: PositionTracker(meas_std=meas_std, gate=gate)
            for label in COLOUR_RANGES
        }

        # Why a colour was not detected in the most recent call(s)
        self.last_reject = {}

    # ----------------------------------------------
    # Capture RGB and depth images
    # ----------------------------------------------

    def capture(self, data):

        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

        if self.depth_noise_std > 0:
            noise = self.rng.normal(0.0, self.depth_noise_std, depth.shape)
            depth = depth + noise.astype(depth.dtype)

        return rgb, depth

    # ----------------------------------------------
    # Geometry helpers
    # ----------------------------------------------

    def _pixel_to_world(self, u, v, d, R_wc, cam_pos):
        """Back-project pixel (u, v) with z-depth d into world coordinates."""
        p_cam = np.array([
            (u - self.cx) * d / self.fx,
            -(v - self.cy) * d / self.fy,
            -d
        ])
        return R_wc @ p_cam + cam_pos

    def _pixel_on_plane(self, u, v, z_plane, R_wc, cam_pos):
        """Intersect the pixel's viewing ray with the horizontal plane Z=z_plane."""
        ray = R_wc @ np.array([
            (u - self.cx) / self.fx,
            -(v - self.cy) / self.fy,
            -1.0
        ])
        if abs(ray[2]) < 1e-9:
            return None
        t = (z_plane - cam_pos[2]) / ray[2]
        if t <= 0:
            return None
        return cam_pos + t * ray

    def _yaw_from_contour(self, contour, z_plane, R_wc, cam_pos):
        """Cube yaw about world Z, folded into [-45, 45) degrees (radians).

        The minimum-area rectangle's corners are projected onto the cube's top
        plane, so the angle is measured in the world horizontal plane rather
        than in the image. Side faces seen at an angle enlarge the contour
        slightly, but the edge directions stay correct.
        """
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        pts = [self._pixel_on_plane(u, v, z_plane, R_wc, cam_pos)
               for u, v in box]
        if pts[0] is None or pts[1] is None:
            return 0.0
        edge = pts[1] - pts[0]
        yaw = np.arctan2(edge[1], edge[0])
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _reject(self, label, reason):
        self.last_reject.setdefault(label, set()).add(reason)

    # ----------------------------------------------
    # Single-frame detection (no filtering)
    # ----------------------------------------------

    def _detect_frame(self, data, rgb, depth):

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        cam_pos = data.cam_xpos[self.cam_id].copy()
        R_wc = data.cam_xmat[self.cam_id].reshape(3, 3).copy()

        xmin, xmax, ymin, ymax = self.pickup_bounds

        detections = {}

        for label, ranges in COLOUR_RANGES.items():

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lo, hi in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.array(lo, dtype=np.uint8),
                    np.array(hi, dtype=np.uint8)
                )

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []

            for contour in contours:

                area = cv2.contourArea(contour)

                if not (self.min_area <= area
                        <= self.width * self.height * 0.10):
                    self._reject(label, 'blob area outside allowed range')
                    continue

                moments = cv2.moments(contour)

                if moments["m00"] <= 0:
                    continue

                u = moments["m10"] / moments["m00"]
                v = moments["m01"] / moments["m00"]

                if self.pixel_noise_std > 0:
                    u += self.rng.normal(0.0, self.pixel_noise_std)
                    v += self.rng.normal(0.0, self.pixel_noise_std)

                selected = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(selected, [contour], -1, 255, -1)

                interior = cv2.erode(selected, self.kernel, iterations=2)
                if not interior.any():
                    interior = selected

                depths = depth[interior > 0]
                valid = depths[np.isfinite(depths) & (depths > 0)]

                if valid.size < 10:
                    self._reject(label, 'too few valid depth pixels')
                    continue

                d = float(np.median(valid))

                p_world = self._pixel_to_world(u, v, d, R_wc, cam_pos)

                if not (xmin <= p_world[0] <= xmax
                        and ymin <= p_world[1] <= ymax):
                    self._reject(label, 'outside pickup bounds')
                    continue

                yaw = self._yaw_from_contour(
                    contour, float(p_world[2]), R_wc, cam_pos)

                candidates.append((area, {
                    "pixel": (float(u), float(v)),
                    "depth_m": d,
                    "position_world_m": p_world.copy(),
                    "yaw_world_rad": yaw,
                    "area_px": float(area),
                    "valid_depth_pixels": int(valid.size),
                    "contour": contour,
                }))

            if candidates:
                detections[label] = max(candidates, key=lambda c: c[0])[1]
            else:
                self._reject(label, 'no candidate blob found')

        return detections

    # ----------------------------------------------
    # Public API
    # ----------------------------------------------

    def detect(self, data, rgb=None, depth=None):
        """One frame, passed through the per-colour tracker."""
        self.last_reject = {}
        if rgb is None or depth is None:
            rgb, depth = self.capture(data)

        found = self._detect_frame(data, rgb, depth)

        for label, det in found.items():
            tracker = self.trackers[label]
            det["position_world_raw_m"] = det["position_world_m"].copy()
            det["position_world_m"] = tracker.update(det["position_world_raw_m"])
            det["position_std_m"] = tracker.std
            det["n_frames"] = 1
        return found

    def detect_averaged(self, data, n_frames=8, min_frames=3):
        """Capture n_frames, average them, then update the tracker once.

        Returns, for each colour seen in at least `min_frames` frames:
          position_world_m      filtered estimate (tracker output)
          position_world_raw_m  plain mean of the frames
          sample_std_m          per-axis spread of the individual frames
          position_std_m        tracker 1-sigma uncertainty
          yaw_world_rad         circular mean of the per-frame yaw (period 90 deg)
          n_frames              frames that contributed
        The remaining keys (pixel, contour, ...) come from the last frame.
        """
        self.last_reject = {}
        samples = {}

        for _ in range(n_frames):
            rgb, depth = self.capture(data)
            for label, det in self._detect_frame(data, rgb, depth).items():
                samples.setdefault(label, []).append(det)

        out = {}
        for label, dets in samples.items():
            n = len(dets)
            if n < min_frames:
                self._reject(label, f'seen in only {n}/{n_frames} frames')
                continue

            P = np.array([d["position_world_m"] for d in dets])
            raw_mean = P.mean(axis=0)
            spread = P.std(axis=0, ddof=1) if n > 1 else np.zeros(3)

            yaws = np.array([d["yaw_world_rad"] for d in dets])
            yaw = float(np.angle(np.mean(np.exp(4j * yaws))) / 4.0)

            tracker = self.trackers[label]
            filtered = tracker.update(raw_mean, meas_var=tracker.r / n)

            det = dict(dets[-1])
            det.update(
                position_world_raw_m=raw_mean,
                position_world_m=filtered,
                position_std_m=tracker.std,
                sample_std_m=spread,
                yaw_world_rad=yaw,
                n_frames=n,
            )
            out[label] = det
        return out


# --------------------------------------------------
# 4. SELECT A TARGET
# --------------------------------------------------

def selected_target(detections, colour):
    label = f"{colour.strip().capitalize()} Cube"
    return detections.get(label)
