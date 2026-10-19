
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

# Pickup workspace in world metres
# xmin, xmax, ymin, ymax
DEFAULT_PICKUP_BOUNDS = (0.24, 0.46, -0.23, 0.23)


# --------------------------------------------------
# 1. LINEAR KALMAN FILTER
# --------------------------------------------------

class LinearKalmanFilter3D:
    """
    Smooths a cube's 3D position.

    State:
        x, y, z = position in metres
        vx, vy, vz = velocity in metres/second

    The camera measures position.
    The filter estimates position and velocity.
    """

    def __init__(self, dt=1.0 / 30.0):
        self.kf = cv2.KalmanFilter(6, 3)

        # Time between measurements
        self.dt = dt

        # Predict the next position using velocity
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Camera measures x, y, z only
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # How noisy the camera measurements are
        self.kf.measurementNoiseCov = (
            np.eye(3, dtype=np.float32) * 0.000025
        )

        # How much the motion model may be wrong
        self.kf.processNoiseCov = (
            np.eye(6, dtype=np.float32) * 0.001
        )

        # Initial uncertainty
        self.kf.errorCovPost = (
            np.eye(6, dtype=np.float32)
        )

        self.initialized = False

    def update(self, position, dt=None):
        """Update from a world XYZ measurement using the actual measurement interval."""
        if dt is not None and np.isfinite(dt) and dt > 1e-4:
            self.dt = float(np.clip(dt, 1e-3, 0.5))
            self.kf.transitionMatrix = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=np.float32)

        position = np.asarray(position, dtype=np.float32).reshape(3, 1)

        # First measurement initializes the filter
        if not self.initialized:
            self.kf.statePost = np.zeros(
                (6, 1), dtype=np.float32
            )
            self.kf.statePost[:3] = position
            self.initialized = True

            return position.ravel().astype(float)

        # Step 1: predict the next position
        self.kf.predict()

        # Step 2: correct the prediction using the camera
        result = self.kf.correct(position)

        # Return the filtered 3D position
        return result[:3, 0].astype(float)

    @property
    def velocity(self):
        """Estimated world velocity in m/s (zero before initialization)."""
        if not self.initialized:
            return np.zeros(3, dtype=float)
        return self.kf.statePost[3:, 0].astype(float).copy()

    def predict_position(self, seconds=0.0):
        """Predict filtered position seconds into the future without changing state."""
        if not self.initialized:
            return None
        seconds = float(max(0.0, seconds))
        p = self.kf.statePost[:3, 0].astype(float)
        v = self.velocity
        return p + v * seconds

    def reset(self):
        """Start tracking again from a new measurement."""
        self.initialized = False
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        self.kf.statePre = np.zeros((6, 1), dtype=np.float32)


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
        min_area=80
    ):
        self.model = model
        self.renderer = renderer
        self.camera_name = camera_name

        self.cam_id = model.camera(camera_name).id

        self.height = int(renderer.height)
        self.width = int(renderer.width)

        # Camera focal length in pixels
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])

        self.fy = self.height / (2.0 * np.tan(fovy / 2.0))
        self.fx = self.fy

        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

        self.pickup_bounds = pickup_bounds
        self.min_area = min_area

        self.kernel = np.ones((3, 3), dtype=np.uint8)

        # Create a separate filter for each cube
        self.position_filters = {
            label: LinearKalmanFilter3D()
            for label in COLOUR_RANGES
        }
        self._last_filter_time = {label: None for label in COLOUR_RANGES}

    # ----------------------------------------------
    # Capture RGB and depth images
    # ----------------------------------------------

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

        return rgb, depth

    # ----------------------------------------------
    # Detect cubes and filter their positions
    # ----------------------------------------------

    def detect(self, data, rgb=None, depth=None):

        if rgb is None or depth is None:
            rgb, depth = self.capture(data)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Current camera position and orientation in world coordinates
        cam_pos = data.cam_xpos[self.cam_id].copy()

        R_wc = data.cam_xmat[self.cam_id].reshape(
            3, 3
        ).copy()

        xmin, xmax, ymin, ymax = self.pickup_bounds

        detections = {}

        # Process each cube colour separately
        for label, ranges in COLOUR_RANGES.items():

            mask = np.zeros(
                hsv.shape[:2], dtype=np.uint8
            )

            # Find pixels matching this colour
            for lo, hi in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.array(lo, dtype=np.uint8),
                    np.array(hi, dtype=np.uint8)
                )

            # Remove noise and fill small gaps
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

                if not (
                    self.min_area <= area
                    <= self.width * self.height * 0.10
                ):
                    continue

                moments = cv2.moments(contour)

                if moments["m00"] <= 0:
                    continue

                # Centre of the detected object in the image
                u = moments["m10"] / moments["m00"]
                v = moments["m01"] / moments["m00"]

                # Use the interior of the contour for depth
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

                valid = depths[
                    np.isfinite(depths) & (depths > 0)
                ]

                if valid.size < 10:
                    continue

                # Median depth reduces the influence of outliers
                d = float(np.median(valid))

                # Convert pixel coordinates into camera coordinates
                p_cam = np.array([
                    (u - self.cx) * d / self.fx,
                    -(v - self.cy) * d / self.fy,
                    -d
                ])

                # Convert camera coordinates into world coordinates
                p_world = R_wc @ p_cam + cam_pos

                # Reject detections outside the pickup workspace
                if not (
                    xmin <= p_world[0] <= xmax
                    and ymin <= p_world[1] <= ymax
                ):
                    continue

                candidates.append((area, {
                    "pixel": (float(u), float(v)),
                    "depth_m": d,
                    "position_world_m": p_world.copy(),
                    "area_px": float(area),
                    "valid_depth_pixels": int(valid.size),
                    "contour": contour,
                }))

            # Keep the largest valid contour for each colour
            if candidates:

                detection = max(
                    candidates,
                    key=lambda item: item[0]
                )[1]

                # Save the original, unfiltered position
                raw_position = detection[
                    "position_world_m"
                ].copy()

                detection["position_world_raw_m"] = (
                    raw_position.copy()
                )

                # Apply this colour's own Kalman filter
                previous_time = self._last_filter_time[label]
                now = float(data.time)
                measured_dt = (now - previous_time) if previous_time is not None else 1.0 / 30.0
                tracker = self.position_filters[label]
                filtered_position = tracker.update(raw_position, measured_dt)
                self._last_filter_time[label] = now

                detection["position_world_m"] = filtered_position
                detection["velocity_world_mps"] = tracker.velocity
                detection["predicted_position_world_m"] = tracker.predict_position(0.10)
                detections[label] = detection

        return detections


# --------------------------------------------------
# 4. SELECT A TARGET
# --------------------------------------------------

def selected_target(detections, colour):
    label = f"{colour.strip().capitalize()} Cube"
    return detections.get(label)
