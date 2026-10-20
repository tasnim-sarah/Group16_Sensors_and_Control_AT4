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
# 1. LINEAR KALMAN FILTER (constant-velocity model, variable time step)
# --------------------------------------------------

class LinearKalmanFilter3D:
    """
    Tracks a cube's 3D position AND velocity.

    State:  [x, y, z, vx, vy, vz]   (metres, metres/second)
    Camera: measures [x, y, z] only.

    Changes vs. the fixed-rate version:
      * dt is passed in on every call (simulation time between frames), so the
        filter stays correct if frames arrive at an irregular rate.
      * the process noise follows the standard "random acceleration" model,
        so a cube that starts/stops/reverses is tracked without big lag.
      * predict() can be used on its own to coast through missed frames.

    Tuning:
      meas_std  - how noisy the camera position is (m). Lower = trust camera.
      accel_std - how hard the cube may accelerate (m/s^2). Higher = reacts
                  faster to starts/stops but gives a noisier velocity.
    """

    def __init__(self, meas_std=0.003, accel_std=0.15):
        self.R = np.eye(3) * meas_std ** 2
        self.q = accel_std ** 2
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False
        self.n_updates = 0

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    def predict(self, dt):
        """Move the state forward by dt seconds (no measurement)."""
        dt = max(float(dt), 0.0)

        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = self.q * dt ** 3 / 3
            Q[i, i + 3] = Q[i + 3, i] = self.q * dt ** 2 / 2
            Q[i + 3, i + 3] = self.q * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, position, dt):
        """Input a measured [x, y, z] taken dt seconds after the previous one;
        return the filtered [x, y, z]."""
        z = np.asarray(position, dtype=float).reshape(3)

        # First measurement initialises the filter
        if not self.initialized:
            self.x = np.zeros(6)
            self.x[:3] = z
            self.P = np.diag([self.R[0, 0]] * 3 + [1.0] * 3)
            self.initialized = True
            self.n_updates = 1
            return z.copy()

        # Step 1: predict
        self.predict(dt)

        # Step 2: correct with the camera measurement
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form
        self.n_updates += 1

        return self.position

    def reset(self):
        """Start tracking again from a new measurement."""
        self.initialized = False
        self.n_updates = 0


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
        max_gap=0.6
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

        # One Kalman filter per cube colour
        self.position_filters = {
            label: LinearKalmanFilter3D()
            for label in COLOUR_RANGES
        }

        # Simulation time of the last accepted measurement per colour
        self.last_meas_time = {label: None for label in COLOUR_RANGES}

        # If a cube is not seen for longer than this, its track is dropped
        self.max_gap = max_gap

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
    # Latest track of one cube (extrapolated to "now")
    # ----------------------------------------------

    def estimate(self, label, now):
        """Filtered position/velocity of a cube extrapolated to time `now`.
        Returns None if the cube has not been seen recently."""

        f = self.position_filters[label]
        last = self.last_meas_time[label]

        if last is None or not f.initialized or now - last > self.max_gap:
            return None

        age = float(now - last)

        return {
            "position": f.position + f.velocity * age,
            "velocity": f.velocity,
            "age": age,
            "n_updates": f.n_updates,}

    # ----------------------------------------------
    # Detect cubes and filter their positions
    # ----------------------------------------------

    def detect(self, data, rgb=None, depth=None):

        if rgb is None or depth is None:
            rgb, depth = self.capture(data)

        now = float(data.time)

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

                # Time since this cube was last measured
                kf = self.position_filters[label]
                last = self.last_meas_time[label]

                if last is not None and now - last > self.max_gap:
                    kf.reset()          # stale track: start again
                    last = None

                dt = 0.0 if last is None else now - last

                # Apply this colour's own Kalman filter
                filtered_position = kf.update(raw_position, dt)

                self.last_meas_time[label] = now

                # Make filtered position + velocity available to the robot
                detection["position_world_m"] = filtered_position
                detection["velocity_world_mps"] = kf.velocity
                detection["n_updates"] = kf.n_updates

                detections[label] = detection

        # Drop tracks of cubes that have not been seen for a while
        for label in COLOUR_RANGES:
            if label in detections:
                continue
            last = self.last_meas_time[label]
            if last is not None and now - last > self.max_gap:
                self.position_filters[label].reset()
                self.last_meas_time[label] = None

        return detections
        



# --------------------------------------------------
# 4. SELECT A TARGET
# --------------------------------------------------

def selected_target(detections, colour):
    label = f"{colour.strip().capitalize()} Cube"
    return detections.get(label)