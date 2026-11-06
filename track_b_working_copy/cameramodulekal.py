
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
        ([22, 180, 150], [38, 255, 255]),
    ],

    "Blue Cube": [
        ([100, 170, 90], [130, 255, 255]),
    ],

    # New: tray colours
    "Pink Tray": [
        ([145, 60, 60], [179, 255, 255]),
    ],

    "Green Tray": [
        ([35, 50, 40], [90, 255, 255]),
    ],

    # Black has low brightness.
    # Shadows and other dark objects may also match.
    "Purple Tray": [
      ([128, 80, 50], [146, 255, 255]),
    ]}


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
            np.eye(3, dtype=np.float32) * meas_noise
        )

        # Motion prediction uncertainty.
        self.kf.processNoiseCov = (
            np.eye(6, dtype=np.float32) * 0.001
        )

        G = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
        self.kf.processNoiseCov = (accel_noise**2 * (G @ G.T)).astype(np.float32)
    

        self.initialized = False

    def update(self, position):
        """Input measured [x, y, z]; return filtered position."""

        position = np.asarray(
            position, dtype=np.float32
        ).reshape(3, 1)

        # Start with the first camera measurement.
        if not self.initialized:
            self.kf.statePost = np.zeros(
                (6, 1), dtype=np.float32
            )

            self.kf.statePost[:3] = position
            self.initialized = True

            return position.flatten().astype(float)

        # Predict the position, then correct it using the camera.
        self.kf.predict()
        result = self.kf.correct(position)

        return result[:3, 0].astype(float)

    def reset(self):
        """Reinitialise when starting a new tracking session."""
        self.initialized = False
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)   # ADD


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
        depth_noise_std=0.0,                                 # ADD
        depth_noise_seed=None,
        accel_noise=0.07,
        meas_noise=2.5e-5,
    ):
        self.model = model
        self.renderer = renderer
        self.camera_name = camera_name

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
        if not np.isfinite(depth_noise_std) or depth_noise_std < 0:   # ADD
            raise ValueError("depth_noise_std must be a finite non-negative value")
        self.depth_noise_std = float(depth_noise_std)
        self.depth_rng = np.random.default_rng(depth_noise_seed)

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
            if self.depth_noise_std > 0.0:                   # ADD
                depth = depth + self.depth_rng.normal(0.0, self.depth_noise_std, depth.shape)

        finally:
            self.renderer.disable_depth_rendering()

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

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []

            for contour in contours:

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

                candidates.append((area, {
                    "name": label,
                    "pixel": (float(u), float(v)),
                    "depth_m": d,
                    "position_world_m": p_world.copy(),
                    "area_px": float(area),
                    "valid_depth_pixels": int(valid.size),
                    "depth_std_m": float(np.std(valid)),     # ADD
                    "contour": contour,
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
                filtered_position = (
                    self.position_filters[label].update(
                        raw_position
                    )
                )

                detection["position_world_m"] = (
                    filtered_position
                )

                detections[label] = detection

        return detections
