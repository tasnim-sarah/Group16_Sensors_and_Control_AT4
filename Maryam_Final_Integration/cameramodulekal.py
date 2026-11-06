
import numpy as np
import cv2
import mujoco


# --------------------------------------------------
# 1. COLOURS TO DETECT
# --------------------------------------------------

COLOUR_RANGES = {
    "Red Cube": [
        ([0, 130, 100], [10, 255, 255]),
        ([170, 130, 100], [179, 255, 255]),
    ],

    "Yellow Cube": [
        ([18, 150, 45], [40, 255, 255]),
    ],

    "Blue Cube": [
        ([95, 140, 70], [130, 255, 255]),
    ],

}
for _colour in ('Red','Yellow','Blue'):
    COLOUR_RANGES[_colour+' Tray'] = COLOUR_RANGES[_colour+' Cube']


# --------------------------------------------------
# 2. SIMPLE 3D KALMAN FILTER
# --------------------------------------------------

class LinearKalmanFilter3D:
    """
    Smooths an object's 3D position.

    State:
        x, y, z = position in metres
        vx, vy, vz = velocity in metres/second
    """

    def __init__(self, dt=1.0 / 30.0, accel_noise=0.07, meas_noise=2.5e-5):
        # accel_noise: std of the object's random acceleration (m/s^2) -> process noise Q
        # meas_noise : variance of one camera position measurement (m^2) -> R
        self.meas_noise = float(meas_noise)
        self.accel_noise = float(accel_noise)
        self.dt = float(dt)
        self.kf = cv2.KalmanFilter(6, 3)

        # Predict position using velocity.
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # The camera measures position, not velocity.
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Camera measurement uncertainty.
        self.kf.measurementNoiseCov = (
            np.eye(3, dtype=np.float32) * self.meas_noise
        )

        # Motion prediction uncertainty.
        # White-noise acceleration: position and velocity noise are tied together.
        eye3 = np.eye(3)
        gain = np.vstack([0.5 * dt**2 * eye3, dt * eye3])
        self.kf.processNoiseCov = (
            float(accel_noise) ** 2 * (gain @ gain.T)
        ).astype(np.float32)

        self.kf.errorCovPost = np.eye(
            6, dtype=np.float32
        )

        self.initialized = False

    def update(self, position, dt=None):
        """Input measured [x, y, z]; return filtered position."""

        interval = self.dt if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("filter interval must be positive")
        transition = np.eye(6, dtype=np.float32)
        transition[:3, 3:] = np.eye(3) * interval
        self.kf.transitionMatrix = transition
        gain = np.vstack([0.5 * interval**2 * np.eye(3), interval * np.eye(3)])
        self.kf.processNoiseCov = (self.accel_noise**2 * gain @ gain.T).astype(np.float32)
        position = np.asarray(
            position, dtype=np.float32
        ).reshape(3, 1)

        # Start with the first camera measurement.
        if not self.initialized:
            self.kf.statePost = np.zeros(
                (6, 1), dtype=np.float32
            )

            self.kf.statePost[:3] = position
            self.kf.errorCovPost = np.diag(
                [self.meas_noise] * 3 + [1.0] * 3
            ).astype(np.float32)
            self.initialized = True

            return position.flatten().astype(float)

        # Predict the position, then correct it using the camera.
        self.kf.predict()
        result = self.kf.correct(position)

        return result[:3, 0].astype(float)

    def reset(self):
        """Reinitialise when starting a new tracking session."""
        self.initialized = False


# --------------------------------------------------
# 3. WRIST CAMERA SETUP
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
    """Calibrate the fixed wrist-camera orientation once during setup only."""

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


def camera_pose_from_joint_positions(
    model,
    data,
    joint_positions,
    camera_name="wrist_cam",
):
    """Compute the wrist camera pose by forward kinematics from estimated joints."""
    q = np.asarray(joint_positions, dtype=float).reshape(6)
    if not np.all(np.isfinite(q)):
        raise ValueError("estimated joint positions must be finite")

    data.qpos[:6] = q
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)

    cam_id = model.camera(camera_name).id
    return (
        data.cam_xpos[cam_id].copy(),
        data.cam_xmat[cam_id].reshape(3, 3).copy(),
    )


# --------------------------------------------------
# 4. CAMERA SENSOR
# --------------------------------------------------

class CameraSensor:

    def __init__(
        self,
        model,
        renderer,
        camera_name="wrist_cam",
        min_area=80,
        max_depth_m=1.5,
        depth_noise_std=0.0,
        depth_noise_seed=None,
        accel_noise=0.07,
        meas_noise=2.5e-5,
        workspace_bounds=None,
        classify_geometry=False,
    ):
        self.model = model
        self.renderer = renderer
        self.camera_name = camera_name
        self.workspace_bounds = None if workspace_bounds is None else np.asarray(workspace_bounds,dtype=float).reshape(3,2)
        self.classify_geometry = bool(classify_geometry)

        self.cam_id = model.camera(camera_name).id

        self.height = int(renderer.height)
        self.width = int(renderer.width)

        # Camera focal length in pixels.
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])

        self.fy = self.height / (2.0 * np.tan(fovy / 2.0))
        self.fx = self.fy

        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

        self.min_area = min_area
        if max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        self.max_depth_m = float(max_depth_m)
        self.kernel = np.ones((3, 3), dtype=np.uint8)

        # Simulated depth sensor noise (m). 0 = noise-free render.
        self.depth_noise_std = float(depth_noise_std)
        self._depth_rng = np.random.default_rng(depth_noise_seed)
        if not np.isfinite(self.depth_noise_std) or self.depth_noise_std < 0:
            raise ValueError("depth_noise_std must be finite and nonnegative")
        self._last_detection_time = {}

        # Each cube and tray gets its own Kalman filter.
        self.position_filters = {
            label: LinearKalmanFilter3D(
                accel_noise=accel_noise, meas_noise=meas_noise
            )
            for label in COLOUR_RANGES
        }

    # --------------------------------------------------
    # Capture RGB and depth images
    # --------------------------------------------------

    def capture(self, data):

        self.renderer.update_scene(
            data, camera=self.camera_name
        )

        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()

        try:
            self.renderer.update_scene(
                data, camera=self.camera_name
            )

            depth = self.renderer.render().copy()

        finally:
            self.renderer.disable_depth_rendering()

        if self.depth_noise_std > 0:
            noise = self._depth_rng.normal(0.0, self.depth_noise_std, depth.shape)
            depth = depth + noise.astype(depth.dtype)

        return rgb, depth

    # --------------------------------------------------
    # Detect cubes and trays
    # --------------------------------------------------

    def detect(
        self,
        data,
        rgb=None,
        depth=None,
        camera_pose=None,
        position_priors=None,
        full_cube_labels=(),
    ):

        if rgb is None or depth is None:
            rgb, depth = self.capture(data)

        if camera_pose is None:
            raise ValueError(
                "camera_pose must be computed from estimated joint positions"
            )

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        cam_pos = np.asarray(camera_pose[0], dtype=float).reshape(3)
        R_wc = np.asarray(camera_pose[1], dtype=float).reshape(3, 3)

        detections = {}

        # Detect each cube and tray separately.
        for label, ranges in COLOUR_RANGES.items():

            mask = np.zeros(
                hsv.shape[:2], dtype=np.uint8
            )

            # Find pixels matching the object's colour.
            for lo, hi in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.array(lo, dtype=np.uint8),
                    np.array(hi, dtype=np.uint8)
                )

            # Remove small noise and fill small gaps.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self.kernel
            )

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self.kernel
            )

            contours, hierarchy = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []

            for contour_index, contour in enumerate(contours):
                # Even nesting levels are coloured regions: a cube inside the
                # neutral tray floor is depth 2. Odd levels are holes, not objects.
                parent = int(hierarchy[0,contour_index,3])
                nesting = 0
                while parent >= 0:
                    nesting += 1
                    parent = int(hierarchy[0,parent,3])
                if nesting % 2:
                    continue

                area = cv2.contourArea(contour)
                x, y, width, height = cv2.boundingRect(contour)
                edge_margin = max(8, round(min(self.width, self.height) * 0.025))
                if (
                    x < edge_margin
                    or y < edge_margin
                    or x + width > self.width - edge_margin
                    or y + height > self.height - edge_margin
                ):
                    continue

                # Close-up cubes can fill a large fraction of the image during grasping.
                max_area = self.width * self.height * 0.35

                if not self.min_area <= area <= max_area:
                    continue

                moments = cv2.moments(contour)

                if moments["m00"] <= 0:
                    continue

                # Centre of the detected object in the image.
                u = moments["m10"] / moments["m00"]
                v = moments["m01"] / moments["m00"]

                # Use the inside of the detected object for depth.
                selected = np.zeros(
                    mask.shape, dtype=np.uint8
                )

                cv2.drawContours(
                    selected, [contour], -1, 255, -1
                )

                interior = cv2.erode(
                    selected, self.kernel, iterations=2
                )

                if not interior.any():
                    interior = selected

                # Use actual colour pixels, rather than filling a tray hole with
                # unrelated floor depth. Thin tray rims retain their colour pixels.
                if label.endswith('Tray'):
                    interior = selected
                interior = cv2.bitwise_and(interior, mask)

                depths = depth[interior > 0]

                # Ignore invalid depth values.
                valid = depths[
                    np.isfinite(depths)
                    & (depths > 0.05)
                    & (depths <= self.max_depth_m)
                ]

                if valid.size < 10:
                    continue

                # Median depth reduces the effect of outliers.
                d = float(np.median(valid))

                # Convert image pixels to camera coordinates.
                p_cam = np.array([
                    (u - self.cx) * d / self.fx,
                    -(v - self.cy) * d / self.fy,
                    -d
                ])

                # Convert camera coordinates to world coordinates.
                p_world = R_wc @ p_cam + cam_pos
                # Reject background candidates BEFORE selecting the largest contour.
                # Otherwise a floor tile can suppress a smaller valid blue cube.
                if self.workspace_bounds is not None:
                    if np.any(p_world < self.workspace_bounds[:,0]) or np.any(p_world > self.workspace_bounds[:,1]):
                        continue

                if self.classify_geometry:
                    yy,xx = np.nonzero((selected>0)&(mask>0)&np.isfinite(depth)&(depth>.05)&(depth<=self.max_depth_m))
                    if len(xx)<10:
                        continue
                    zz = depth[yy,xx]
                    cloud_camera = np.column_stack(((xx-self.cx)*zz/self.fx,
                        -(yy-self.cy)*zz/self.fy,-zz))
                    cloud_world = cloud_camera @ R_wc.T + cam_pos
                    low,high = np.percentile(cloud_world,[1,99],axis=0)
                    extent = high[:2]-low[:2]
                    if label.endswith('Cube'):
                        # A finger can hide most of one cube face during the
                        # verification lift. Keep compact partial faces, while
                        # rejecting long thin rim segments.
                        if (np.min(extent)<.004 or np.max(extent)>.075
                            or np.max(extent)<.018
                            or np.max(extent)/max(np.min(extent),1e-9)>5):
                            continue
                        if label in full_cube_labels and np.min(extent)<.028:
                            continue
                    else:
                        if np.min(extent)<.075 or np.max(extent)>.17:
                            continue
                        p_world[:2] = (low[:2]+high[:2])*.5

                candidates.append((area, {
                    "name": label,
                    "pixel": (float(u), float(v)),
                    "depth_m": d,
                    "position_world_m": p_world.copy(),
                    "area_px": float(area),
                    "valid_depth_pixels": int(valid.size),
                    "depth_std_m": float(np.std(valid)),
                    "contour": contour,
                    "classification": 'cube' if label.endswith('Cube') else 'tray',
                }))

            # Keep the strongest cube detection. For trays, a supplied static
            # location prior associates separated coloured wall components with
            # the correct tray; the returned position remains RGB-D measured.
            if candidates:
                if position_priors and label in position_priors:
                    prior_xy = np.asarray(position_priors[label], dtype=float)[:2]
                    detection = min(
                        candidates,
                        key=lambda item: np.linalg.norm(
                            item[1]["position_world_m"][:2] - prior_xy
                        ),
                    )[1]
                else:
                    detection = max(
                        candidates,
                        key=lambda item: item[0]
                    )[1]

                # Save the original camera measurement.
                raw_position = detection[
                    "position_world_m"
                ].copy()

                detection["position_world_raw_m"] = (
                    raw_position.copy()
                )

                # Smooth the position with its own Kalman filter.
                previous_time = self._last_detection_time.get(label)
                interval = None if previous_time is None else max(1e-6, float(data.time)-previous_time)
                self._last_detection_time[label] = float(data.time)
                filtered_position = (
                    self.position_filters[label].update(
                        raw_position, dt=interval
                    )
                )

                detection["position_world_m"] = (
                    filtered_position
                )

                detections[label] = detection

        return detections
