"""
Sorting robot: UR3e arm + wrist camera + gripper (MuJoCo).
The camera finds the coloured cubes and trays, the arm picks each cube in the
order you type and drops it into its matching tray.
Run with:  python test.py
"""
import csv
import os
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from cameramodulekal import (
    CameraSensor,
    add_wrist_camera,
    aim_camera_once,
    camera_pose_from_joint_positions,
)
from controlfile import (
    JointKalmanObserver,
    PositionActuatorLQRController,
    PositionActuatorStateSpaceModel,
    tracking_metrics,
)
from vision_monitor import VisionMonitor

# --------------------------------------------------
# SETTINGS (change these to play around)
# --------------------------------------------------
HOVER_Z = 0.12            # height of the gripper tip when travelling (m)
GRASP_Z = 0.03            # height of the gripper tip when grabbing a cube (m)
DROP_Z = 0.12             # height of the gripper tip when releasing over a tray (m)
                          # (tray walls are 0.05 m tall and the cube hangs ~0.03 m below
                          #  the TCP, so keep this at 0.10 or higher)
FINGER_TRAVEL = 0.034     # how far each finger slides when closing (m)
FINGER_OFFSET = 0.045     # distance of each finger from the centre when open (m)
GRIPPER_POSITION_KP = 1500.0
GRIPPER_OPEN = 0.0        # gripper command: open
GRIPPER_CLOSED = FINGER_TRAVEL   # gripper command: closed
MAX_SPEED = 0.10          # conservative Cartesian speed for carrying a cube (m/s)
PBVS_LOOKAHEAD_M = MAX_SPEED * 0.2
PAN_SPEED = 0.4           # speed the base turns towards a tray (rad/s)
CUBE_HALF = 0.02          # half a cube's size (m): the camera sees the front face, so we aim this far deeper
RED_AMPLITUDE = 0.06      # the red cube slides this far left and right (m)
RED_PERIOD = 6.0          # seconds for the red cube to slide there and back
LEAD_TIME = 0.15          # aim this far ahead of the moving red cube (s)
PRINT_EVERY = 6           # print the red cube every 6th camera picture (6 x 1/30 s = 0.2 s)

# --- tray placement tuning ---
TRAY_XY_CLAMP = 0.02            # trust the camera only within +/- this of the nominal tray xy (m)
TRAY_PLAUSIBLE_RADIUS = 0.05    # reject tray detections further than this from nominal (m)
RELEASE_SETTLE_S = 0.3          # pause over the tray before opening (s)
RELEASE_RAMP_S = 0.4            # time taken to open the gripper gradually (s)
COMPENSATE_HELD_OFFSET = False  # DIAGNOSTIC ONLY: uses simulator truth for the cube-in-gripper offset

LOOK_CUBES = np.array([0.35, 0.0, 0.03])   # where the camera looks at the start (x, y, z in m)
VISION_MONITOR_ENABLED = True
CAMERA_RENDER_WIDTH = 320
CAMERA_RENDER_HEIGHT = 240
IK_POSITION_TOLERANCE = 0.010
IK_ROTATION_TOLERANCE = 0.02
TRAY_MEASUREMENT_TOLERANCE = 0.025

CUBES = ["Red Cube", "Yellow Cube", "Blue Cube"]
CUBE_TO_TRAY = {"Red Cube": "Pink Tray", "Yellow Cube": "Green Tray", "Blue Cube": "Purple Tray"}
TRAY_BODY = {"Pink Tray": "red_bin", "Green Tray": "yellow_bin", "Purple Tray": "blue_bin"}
TRAY_XY = {"Pink Tray": (0.32, 0.28), "Green Tray": (0.00, 0.40), "Purple Tray": (0.16, 0.36)}
NAMES = {
    "r": "Red Cube", "red": "Red Cube",
    "y": "Yellow Cube", "yellow": "Yellow Cube",
    "b": "Blue Cube", "blue": "Blue Cube",
    "g": "Green Cube", "green": "Green Cube",
}


def parse_pick_order(text, available_cubes):
    """Parse comma/space-separated colour names and abbreviations."""
    tokens = [token.lower() for token in text.replace(",", " ").split()]
    if not tokens:
        raise ValueError("enter at least one cube colour")

    unknown = [token for token in tokens if token not in NAMES]
    if unknown:
        raise ValueError(f"unknown colour(s): {', '.join(unknown)}")

    order = [NAMES[token] for token in tokens]
    if len(set(order)) != len(order):
        raise ValueError("each cube can only be requested once")

    unavailable = [cube for cube in order if cube not in available_cubes]
    if unavailable:
        names = ", ".join(cube.removesuffix(" Cube") for cube in unavailable)
        raise ValueError(
            f"{names} cube(s) are not present in this scene; "
            "available cubes are red, blue and yellow"
        )
    return order

# Rough tray positions (x, y, z), built from TRAY_XY. Only used as a first guess.
BIN_AREA = {tray: np.array([x, y, 0.02]) for tray, (x, y) in TRAY_XY.items()}

# --------------------------------------------------
# 1. BUILD THE SCENE
# --------------------------------------------------
spec = mujoco.MjSpec.from_file(str(Path(__file__).resolve().parent / "universal_robots_ur3e" / "pick_scene.xml"))
for key in list(spec.keys):      # delete saved keyframes, because we add a gripper to the model
    spec.delete(key)
add_wrist_camera(spec)           # from cameramodulekal.py


def add_gripper(spec):
    """Add a two-finger gripper to the last arm link."""
    wrist = spec.body("wrist_3_link")

    # a small dark box that represents the camera body
    wrist.add_geom(name="camera_shell", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.045, 0.0125, 0.0125], pos=[0, 0.08, 0.05],
                   rgba=[0.15, 0.15, 0.15, 1], mass=0.072, contype=0, conaffinity=0)

    # the gripper base, turned so the fingers point down
    gripper = wrist.add_body(name="gripper_base", pos=[0, 0.09215, 0])
    gripper.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    gripper.alt.xyaxes = [0, 0, 1, 1, 0, 0]

    # the palm
    gripper.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.045, 0.01],
                     pos=[0, 0, 0.01], mass=0.25, rgba=[0.2, 0.2, 0.2, 1], contype=1, conaffinity=1)

    # two fingers; each one slides towards the middle
    for side, sign in [("left", 1), ("right", -1)]:
        finger = gripper.add_body(name=f"finger_{side}", pos=[0, sign * FINGER_OFFSET, 0.04])
        finger.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                         axis=[0, -sign, 0], range=[0, FINGER_TRAVEL], damping=5, armature=0.001)
        finger.add_geom(name=f"finger_{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.008, 0.005, 0.02], mass=0.05, rgba=[0.3, 0.3, 0.3, 1],
                        friction=[1.5, 0.01, 0.001], condim=4, contype=1, conaffinity=1)

    # the tool centre point (tcp): the spot between the fingertips that we move around
    gripper.add_site(name="tcp", pos=[0, 0, 0.045], size=[0.006, 0, 0], rgba=[1, 0, 1, 1])

    # the right finger always copies the left finger, so one motor closes both
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      name1="finger_right", name2="finger_left")
    motor = spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    motor.set_to_position(kp=GRIPPER_POSITION_KP)
    motor.ctrlrange = [0, FINGER_TRAVEL]
    motor.forcerange = [-30, 30]


add_gripper(spec)
model = spec.compile()           # turn the description into a model
data = mujoco.MjData(model)      # data holds the live state (joint angles, positions, ...)

# start with the arm in its HOME pose
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME

# move the trays to the spots in TRAY_XY
for tray, (x, y) in TRAY_XY.items():
    model.body_pos[model.body(TRAY_BODY[tray]).id][:2] = [x, y]

# put the cubes at random spots. Each cube gets its own strip of y values (side to side),
# and which colour gets which strip is shuffled, so the cubes never touch.
cube_seed = os.environ.get("CUBE_SEED")
random_seed = None if cube_seed is None else int(cube_seed)
rng = np.random.default_rng(random_seed)
strips = [(0.10, 0.15), (-0.025, 0.025), (-0.15, -0.10)]
rng.shuffle(strips)
for joint, (ymin, ymax) in zip(["red_cube_joint", "yellow_cube_joint", "blue_cube_joint"], strips):
    adr = int(model.jnt_qposadr[model.joint(joint).id])   # where this joint starts in data.qpos
    data.qpos[adr] = rng.uniform(0.28, 0.35)              # x (forwards)
    data.qpos[adr + 1] = rng.uniform(ymin, ymax)          # y (sideways)

mujoco.mj_forward(model, data)                            # update everything after our changes
camera_alignment = aim_camera_once(model, data, LOOK_CUBES)  # calibrate once at the home pose

# remember a few things about the arm
BASE_XY = np.zeros(2)
site_id = model.site("tcp").id
FINGER_ADR = int(model.jnt_qposadr[model.joint("finger_left").id])

fk_data = mujoco.MjData(model)


def forward_kinematics(joint_positions):
    """Return the TCP pose using only the supplied six-joint configuration."""
    fk_data.qpos[:6] = np.asarray(joint_positions, dtype=float).reshape(6)
    fk_data.qvel[:6] = 0.0
    mujoco.mj_forward(model, fk_data)
    return (
        fk_data.site_xpos[site_id].copy(),
        fk_data.site_xmat[site_id].reshape(3, 3).copy(),
    )


P_HOME, R_HOME = forward_kinematics(HOME)
R_des = R_HOME.copy()
camera_id = model.camera("wrist_cam").id
R_TCP_CAMERA = R_HOME.T @ fk_data.cam_xmat[camera_id].reshape(3, 3)


def camera_rotation_toward(camera_position, target_position):
    """Return a world camera rotation whose optical axis faces the target."""
    forward = np.asarray(target_position, dtype=float).reshape(3) - np.asarray(
        camera_position, dtype=float
    ).reshape(3)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("camera and tray search point are too close")
    z_axis = -forward / norm
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, z_axis))) > 0.98:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))

# the red cube slides, so remember its joint and its starting y
red_joint = model.joint("red_cube_joint")
red_qadr = int(model.jnt_qposadr[red_joint.id])
red_dadr = int(model.jnt_dofadr[red_joint.id])
red_y0 = float(data.qpos[red_qadr + 1])
red_moving = True

q_reference = HOME.copy()
gripper_cmd = GRIPPER_OPEN       # gripper command we ask for
dt = model.opt.timestep          # length of one physics step (s)
DETECT_EVERY = round(1 / 30 / dt)   # take a camera picture every this many steps (30 per second)

ENCODER_NOISE_STD = float(os.environ.get("ENCODER_NOISE_STD", "5e-4"))
if not np.isfinite(ENCODER_NOISE_STD) or ENCODER_NOISE_STD < 0:
    raise ValueError("ENCODER_NOISE_STD must be a finite non-negative value")
encoder_rng = np.random.default_rng(
    None if random_seed is None else random_seed + 1
)
control_model = PositionActuatorStateSpaceModel(model, data)
control_model.print_report()
controller = PositionActuatorLQRController(control_model)
observer = JointKalmanObserver(
    control_model,
    position_noise=ENCODER_NOISE_STD**2,
    process_noise=1e-5,
)
q_est, dq_est = observer.reset(
    data.qpos[:6] + encoder_rng.normal(0.0, ENCODER_NOISE_STD, 6)
)
previous_control = data.ctrl[:6].copy()
motion_target = None

joint_reference_samples = []
joint_truth_samples = []
joint_estimate_samples = []
joint_sample_times = []
model_prediction_residuals = []
away_from_home_samples = []
camera_position_errors = []
placement_results = []
pick_results = []
task_results = []
last_failure_recoverable = False
pbvs_position_errors = []
object_localization_errors = {name: [] for name in (*CUBES, *TRAY_BODY.keys())}
rejected_detection_counts = {name: 0 for name in object_localization_errors}
localization_outlier_counts = {name: 0 for name in object_localization_errors}
actuator_saturation_counts = np.zeros(6, dtype=int)
last_ik_diagnostics = None
last_vision_frames = None
task_stage = "Waiting for pick order"
vision_capture_times = []
vision_detection_times = []
vision_display_times = []

# --------------------------------------------------
# 2. CAMERA TRACKING
# --------------------------------------------------
renderer = mujoco.Renderer(
    model, width=CAMERA_RENDER_WIDTH, height=CAMERA_RENDER_HEIGHT
)
sensor = CameraSensor(model, renderer)   # from cameramodulekal.py: finds colours + Kalman filter
vision_monitor = VisionMonitor(
    sensor.width, sensor.height, max_depth_m=sensor.max_depth_m
)
camera_id = model.camera("wrist_cam").id
print(
    "Camera calibration: "
    f"fx={sensor.fx:.1f}px fy={sensor.fy:.1f}px "
    f"cx={sensor.cx:.1f}px cy={sensor.cy:.1f}px; "
    f"wrist-camera offset={np.round(model.cam_pos[camera_id], 4)}m; "
    f"initial alignment={camera_alignment:.4f}"
)

tracks = {}      # name -> last known position of that object (x, y, z)
velocity = {}    # name -> speed of that object from the Kalman filter
track_last_seen = {}
tray_anchors = {}
target = None    # the object we are working on now: a cube name or a tray name
visible_targets = set()
rough_fallback_targets = set()


def update_tracks():
    """Localise RGB-D detections with the camera pose from estimated-joint FK."""
    global visible_targets, last_vision_frames
    estimated_camera_pose = camera_pose_from_joint_positions(
        model, fk_data, q_est
    )
    # Evaluation only; this true pose is not passed to image localization or control.
    camera_position_errors.append(
        float(np.linalg.norm(data.cam_xpos[sensor.cam_id] - estimated_camera_pose[0]))
    )
    capture_started = time.perf_counter()
    rgb, depth = sensor.capture(data)
    vision_capture_times.append(time.perf_counter() - capture_started)
    detection_started = time.perf_counter()
    found = sensor.detect(
        data,
        rgb,
        depth,
        camera_pose=estimated_camera_pose,
        position_priors=BIN_AREA,
    )
    for name in list(found):
        raw_position = found[name]["position_world_raw_m"]
        if (
            name in TRAY_BODY
            and name in tray_anchors
            and np.linalg.norm(
                raw_position[:2] - tray_anchors[name][:2]
            ) > TRAY_MEASUREMENT_TOLERANCE
        ):
            rejected_detection_counts[name] += 1
            tray_filter = sensor.position_filters[name]
            tray_filter.reset()
            tray_filter.update(tray_anchors[name])
            del found[name]
            continue
        if name in CUBES:
            plausible = (
                0.08 <= raw_position[0] <= 0.50
                and -0.30 <= raw_position[1] <= 0.50
                and -0.03 <= raw_position[2] <= 0.10
            )
        else:
            center = np.asarray(TRAY_XY[name], dtype=float)
            plausible = (
                np.linalg.norm(raw_position[:2] - center) <= TRAY_PLAUSIBLE_RADIUS
                and -0.05 <= raw_position[2] <= 0.12
            )
        if not plausible:
            rejected_detection_counts[name] += 1
            sensor.position_filters[name].reset()
            del found[name]
    vision_detection_times.append(time.perf_counter() - detection_started)
    last_vision_frames = {"rgb": rgb, "depth": depth, "detections": found}
    visible_targets = set(found)
    for name in found:
        rough_fallback_targets.discard(name)
        track_last_seen[name] = data.time
        pos = found[name]["position_world_m"].copy()
        if name in CUBES:
            ray = pos - estimated_camera_pose[0]
            pos = pos + CUBE_HALF * ray / np.linalg.norm(ray)
            body_name = name.lower().replace(" ", "_")
        else:
            body_name = TRAY_BODY.get(name)
        if body_name is not None:
            # Evaluation only; simulator object positions never enter the target.
            object_truth = data.xpos[model.body(body_name).id]
            localization_error = float(np.linalg.norm(pos - object_truth))
            object_localization_errors[name].append(localization_error)
            if localization_error > 0.05:
                localization_outlier_counts[name] += 1
        tracks[name] = pos
        velocity[name] = sensor.position_filters[name].kf.statePost[3:6, 0].astype(float)
    for name in tracks:
        if name not in found and name not in rough_fallback_targets:
            if name == "Red Cube" and red_moving:
                filter_state = sensor.position_filters[name].kf.statePost[:, 0]
                velocity[name] = filter_state[3:6].astype(float)
                elapsed = data.time - track_last_seen[name]
                tracks[name] = (
                    filter_state[:3].astype(float) + velocity[name] * elapsed
                )
    return found


def target_tracking_usable():
    """Allow brief camera occlusion while rejecting stale targets."""
    if target in tracks and (target != "Red Cube" or not red_moving):
        return True
    return (
        target is None
        or target in visible_targets
        or target in rough_fallback_targets
        or data.time - track_last_seen.get(target, -np.inf) <= 3.0
    )


def cube_xy(name):
    """x, y where the gripper should go for this cube."""
    pos = tracks[name]
    if name == "Red Cube" and red_moving:
        pos = pos + velocity[name] * LEAD_TIME        # aim a little ahead of the moving red cube
    return pos[:2]


def tray_xy(tray):
    """Measured tray centre, clamped to +/- TRAY_XY_CLAMP of the nominal position.

    The trays are fixed, so the camera is only trusted for small corrections."""
    nominal = np.asarray(TRAY_XY[tray], dtype=float)
    offset = np.clip(tracks[tray][:2] - nominal, -TRAY_XY_CLAMP, TRAY_XY_CLAMP)
    return nominal + offset


# --------------------------------------------------
# 3. MOVING THE ARM
# --------------------------------------------------
def rotation_vector(rotation):
    """Convert a small world-frame rotation matrix to its axis-angle vector."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1e-7:
        return 0.5 * skew
    return angle * skew / (2.0 * np.sin(angle))


def inverse_kinematics(target_position, target_rotation, q_seed):
    """Solve a TCP pose target and report convergence and residual errors."""
    q = np.asarray(q_seed, dtype=float).reshape(6).copy()
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    limits = model.jnt_range[:6]
    epsilon = 1e-5
    orientation_weight = 0.35

    for _ in range(80):
        position, rotation = forward_kinematics(q)
        position_error = target_position - position
        rotation_error = rotation_vector(target_rotation @ rotation.T)
        error = np.r_[position_error, orientation_weight * rotation_error]
        if (
            np.linalg.norm(position_error) < IK_POSITION_TOLERANCE
            and np.linalg.norm(rotation_error) < IK_ROTATION_TOLERANCE
        ):
            return q, True, float(np.linalg.norm(position_error)), float(
                np.linalg.norm(rotation_error)
            )

        jacobian = np.zeros((6, 6))
        for joint in range(6):
            perturbed = q.copy()
            perturbed[joint] += epsilon
            p_next, r_next = forward_kinematics(perturbed)
            jacobian[:3, joint] = (p_next - position) / epsilon
            jacobian[3:, joint] = (
                orientation_weight
                * rotation_vector(r_next @ rotation.T)
                / epsilon
            )

        damping = 0.025
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6),
            error,
        )
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0.2:
            delta *= 0.2 / delta_norm
        q = np.clip(q + delta, limits[:, 0], limits[:, 1])

    position, rotation = forward_kinematics(q)
    position_error = float(np.linalg.norm(target_position - position))
    rotation_error = float(
        np.linalg.norm(rotation_vector(target_rotation @ rotation.T))
    )
    return (
        q,
        position_error < IK_POSITION_TOLERANCE
        and rotation_error < IK_ROTATION_TOLERANCE,
        position_error,
        rotation_error,
    )


def update_pbvs_reference():
    """Turn the latest filtered RGB-D target into a short Cartesian IK waypoint."""
    global last_ik_diagnostics
    point = np.asarray(motion_target(), dtype=float).reshape(3)
    tcp_position, _ = forward_kinematics(q_est)
    delta = point - tcp_position
    distance = np.linalg.norm(delta)
    max_step = PBVS_LOOKAHEAD_M
    waypoint = (
        point if distance <= max_step
        else tcp_position + delta * (max_step / distance)
    )
    q_solution, converged, position_error, rotation_error = inverse_kinematics(
        waypoint, R_des, q_est
    )
    last_ik_diagnostics = {
        "target": point.copy(),
        "waypoint": waypoint.copy(),
        "converged": converged,
        "position_error": position_error,
        "rotation_error": rotation_error,
    }
    if converged:
        q_reference[:] = q_solution
    return converged


def wrap(angle):
    """Turn any angle into the range -pi to +pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def azimuth(xy):
    """The angle of a point as seen from the base."""
    d = np.asarray(xy) - BASE_XY
    return np.arctan2(d[1], d[0])


step = 0


def update_vision_monitor():
    if not VISION_MONITOR_ENABLED or not vision_monitor.enabled or last_vision_frames is None:
        return

    detection = last_vision_frames["detections"].get(target)
    if target is None:
        target_status = "IDLE"
    elif target in rough_fallback_targets:
        target_status = "FALLBACK"
    elif target in visible_targets:
        target_status = "VISIBLE"
    elif target in tracks and data.time - track_last_seen.get(target, -np.inf) <= 3.0:
        target_status = (
            "PREDICTED"
            if target == "Red Cube" and red_moving
            else "OCCLUDED"
        )
    else:
        target_status = "LOST"

    raw_position = (
        None if detection is None else detection["position_world_raw_m"]
    )
    filtered_position = (
        detection["position_world_m"]
        if detection is not None
        else tracks.get(target) if target is not None else None
    )
    pbvs_target = None
    pbvs_error = None
    if motion_target is not None:
        pbvs_target = np.asarray(motion_target(), dtype=float).reshape(3)
        tcp_position, _ = forward_kinematics(q_est)
        pbvs_error = float(np.linalg.norm(pbvs_target - tcp_position))

    monitor_started = time.perf_counter()
    vision_monitor.update(
        last_vision_frames,
        target_name=target,
        target_status=target_status,
        raw_position=raw_position,
        filtered_position=filtered_position,
        pbvs_target=pbvs_target,
        pbvs_error=pbvs_error,
        stage=task_stage,
        q_estimated=q_est,
        q_ground_truth=data.qpos[:6],
        observer_std=observer.position_standard_deviation(),
        lqr_active=controller is not None,
    )
    vision_display_times.append(time.perf_counter() - monitor_started)


def step_sim():
    """Run one physics step with observer feedback and six-joint position LQR."""
    global step, q_est, dq_est, previous_control
    if not viewer.is_running():
        raise SystemExit

    if red_moving:
        w = 2 * np.pi / RED_PERIOD
        data.qpos[red_qadr + 1] = red_y0 + RED_AMPLITUDE * np.sin(w * data.time)
        data.qvel[red_dadr + 1] = RED_AMPLITUDE * w * np.cos(w * data.time)
        mujoco.mj_forward(model, data)

    encoder_q = data.qpos[:6] + encoder_rng.normal(0.0, ENCODER_NOISE_STD, 6)
    q_est, dq_est = observer.update(encoder_q, previous_control)

    if step % DETECT_EVERY == 0:
        camera_frame_started = time.perf_counter()
        found = update_tracks()
        if motion_target is not None and target_tracking_usable():
            update_pbvs_reference()
        if red_moving and (step // DETECT_EVERY) % PRINT_EVERY == 0:
            if "Red Cube" in found:
                red = found["Red Cube"]
                print(f"t={data.time:5.2f}s  camera {np.round(red['position_world_raw_m'], 3)}  "
                      f"kalman {np.round(red['position_world_m'], 3)}  "
                      f"velocity {np.round(velocity['Red Cube'], 3)}")
            else:
                print(f"t={data.time:5.2f}s  Red Cube not in view")
        viewer.sync()
        update_vision_monitor()
        remaining_frame_time = 1 / 30 - (
            time.perf_counter() - camera_frame_started
        )
        if remaining_frame_time > 0:
            time.sleep(remaining_frame_time)

    raw_position_command = controller.raw_command(q_reference, q_est, dq_est)
    actuator_saturation_counts[:] += (
        (raw_position_command < control_model.ctrl_limits[:, 0])
        | (raw_position_command > control_model.ctrl_limits[:, 1])
    )
    position_command = controller.command(q_reference, q_est, dq_est)
    data.ctrl[:6] = position_command
    previous_control = position_command.copy()
    data.ctrl[6] = gripper_cmd                        # send the gripper command
    joint_reference_samples.append(q_reference.copy())
    joint_truth_samples.append(data.qpos[:6].copy())
    joint_estimate_samples.append(q_est.copy())
    joint_sample_times.append(float(data.time))
    state_before_step = np.r_[data.qpos[:6], data.qvel[:6]].copy()
    control_before_step = position_command.copy()
    mujoco.mj_step(model, data)
    predicted_state = (
        control_model.A @ state_before_step
        + control_model.B @ control_before_step
        + control_model.offset
    )
    actual_state = np.r_[data.qpos[:6], data.qvel[:6]]
    model_prediction_residuals.append(
        float(np.max(np.abs(actual_state - predicted_state)))
    )
    away_from_home_samples.append(bool(np.linalg.norm(q_reference - HOME) > 0.1))
    step += 1


def goto(get_target, tol=0.01, carry_check=None):
    """Follow a vision-updated PBVS target; return whether it was reached."""
    global motion_target, last_ik_diagnostics
    motion_target = get_target
    last_ik_diagnostics = None
    start = data.time
    initial_tcp_position, _ = forward_kinematics(q_est)
    initial_target = np.asarray(get_target(), dtype=float).reshape(3)
    timeout = max(
        12.0,
        2.0 * float(np.linalg.norm(initial_target - initial_tcp_position))
        / MAX_SPEED
        + 4.0,
    )
    settled_steps = 0
    last_report = start
    last_carry_check = start
    success = False
    while data.time - start < timeout:
        point = np.asarray(motion_target(), dtype=float).reshape(3)
        tcp_position, _ = forward_kinematics(q_est)
        tcp_error = float(np.linalg.norm(tcp_position - point))
        if target_tracking_usable() and tcp_error < tol:
            settled_steps += 1
            if settled_steps >= max(1, round(0.1 / dt)):
                success = True
                break
        else:
            settled_steps = 0
        step_sim()
        if carry_check is not None and data.time - last_carry_check >= 0.02:
            if not carry_check("in-transit carry check"):
                print("  Carry check failed during PBVS transit")
                break
            last_carry_check = data.time
        if data.time - last_report >= 0.2:
            true_joint_error = float(np.max(np.abs(data.qpos[:6] - q_reference)))
            estimated_joint_error = float(np.max(np.abs(q_est - q_reference)))
            raw_position_command = controller.raw_command(q_reference, q_est, dq_est)
            saturated = np.flatnonzero(
                (raw_position_command < control_model.ctrl_limits[:, 0])
                | (raw_position_command > control_model.ctrl_limits[:, 1])
            ) + 1
            ik_report = last_ik_diagnostics or {}
            print(
                f"  State feedback t={data.time:.2f}s "
                f"target={np.round(point, 3)} TCP_est={np.round(tcp_position, 3)} "
                f"TCP_err={tcp_error:.3f}m "
                f"joint_est_err={estimated_joint_error:.3f}rad "
                f"joint_truth_err={true_joint_error:.3f}rad "
                f"target_visible={target in visible_targets} "
                f"track_age={data.time - track_last_seen.get(target, data.time):.2f}s "
                f"IK_pos_err={ik_report.get('position_error', float('nan')):.4f}m "
                f"IK_rot_err={ik_report.get('rotation_error', float('nan')):.3f}rad "
                f"q_ref={np.round(q_reference, 2)} "
                f"q_est={np.round(q_est, 2)} "
                f"u_pos={np.round(data.ctrl[:6], 2)} "
                f"saturated_joints={saturated.tolist()}"
            )
            last_report = data.time
        if last_ik_diagnostics is not None and not last_ik_diagnostics["converged"]:
            print(
                "  IK failed: "
                f"position residual={last_ik_diagnostics['position_error']:.4f}m, "
                f"orientation residual={last_ik_diagnostics['rotation_error']:.3f}rad"
            )
            break
    else:
        print(
            f"  PBVS move timed out after {timeout:.1f}s: "
            f"target={np.round(point, 3)}, TCP={np.round(tcp_position, 3)}, "
            f"position error={tcp_error:.3f}m"
        )
    motion_target = None
    if success:
        pbvs_position_errors.append(tcp_error)
        print(
            f"  PBVS reached target: desired={np.round(point, 3)} "
            f"TCP_est={np.round(tcp_position, 3)} error={tcp_error:.4f}m"
        )
    return success


def wait(seconds):
    """Hold the gripper still for a while."""
    start = data.time
    while data.time - start < seconds:
        step_sim()


def release_gripper(duration=RELEASE_RAMP_S):
    """Ramp the gripper open instead of snapping it, so the cube is not flicked."""
    global gripper_cmd
    start, start_cmd = data.time, gripper_cmd
    while data.time - start < duration:
        a = (data.time - start) / duration
        gripper_cmd = start_cmd * (1.0 - a)      # GRIPPER_OPEN is 0
        step_sim()
    gripper_cmd = GRIPPER_OPEN


def rotate_base(pan_target):
    """Pan the robot with LQR and retain the new estimated tool orientation."""
    global R_des
    while abs(wrap(pan_target - q_reference[0])) > 0.01:
        q_reference[0] += np.clip(
            wrap(pan_target - q_reference[0]),
            -PAN_SPEED * dt,
            PAN_SPEED * dt,
        )
        step_sim()
    start = data.time
    while data.time - start < 0.5:
        step_sim()
    _, R_des = forward_kinematics(q_est)


# --------------------------------------------------
# 4. PICK AND PLACE
# --------------------------------------------------
def pick_and_place(cube, tray):
    global gripper_cmd, red_moving, target, R_des, task_stage
    global last_failure_recoverable
    picked = False
    placed = False
    recorded = False
    last_failure_recoverable = False

    def finish(sequence_success):
        nonlocal recorded
        if not recorded:
            pick_results.append(picked)
            placement_results.append(placed)
            recorded = True
        return bool(sequence_success)

    def abort(reason, recoverable=False):
        global last_failure_recoverable
        last_failure_recoverable = recoverable
        print(f"  Aborting {cube}: {reason}")
        return finish(False)

    def return_home_after_early_release():
        global target, R_des, task_stage
        task_stage = "Returning home after verified tray deposit"
        target = None
        R_des = R_HOME.copy()
        if not goto(lambda: P_HOME, tol=0.02):
            return abort("return-home move failed after the verified tray deposit")
        rotate_base(HOME[0])
        R_des = R_HOME.copy()
        for name in CUBES:
            tracks.pop(name, None)
            track_last_seen.pop(name, None)
            sensor.position_filters[name].reset()
        task_stage = "Task complete"
        print(f"<<< {cube} done")
        return finish(True)

    def accept_early_release():
        nonlocal placed
        global gripper_cmd, target, task_stage
        if not carry_failure_detected:
            return False
        task_stage = "Verifying destination after carry loss"
        target = None
        gripper_cmd = GRIPPER_OPEN
        wait(0.7)
        cube_position = data.xpos[model.body(cube_body).id].copy()
        tray_position = data.xpos[model.body(TRAY_BODY[tray]).id].copy()
        in_tray = (
            np.all(np.abs(cube_position[:2] - tray_position[:2]) <= 0.055)
            and 0.0 <= cube_position[2] <= 0.12
        )
        print(
            "  UNPLANNED RELEASE CHECK: "
            f"{'cube settled in destination' if in_tray else 'cube outside destination'} "
            f"(cube centre {np.round(cube_position, 3)})"
        )
        if in_tray:
            placed = True
            print("  PLACEMENT CHECK: PASS (cube settled in the verified destination)")
            return True
        return False

    def handle_carry_failure(reason):
        if accept_early_release():
            return return_home_after_early_release()
        if carry_failure_detected:
            return abort("cube was lost outside the destination during transfer", True)
        return abort(reason)

    def ensure_tray_localized():
        global target, R_des, task_stage
        target = None
        tracks.pop(tray, None)
        track_last_seen.pop(tray, None)
        sensor.position_filters[tray].reset()
        camera_position, _ = camera_pose_from_joint_positions(
            model, fk_data, q_est
        )
        R_des = camera_rotation_toward(
            camera_position, BIN_AREA[tray]
        ) @ R_TCP_CAMERA.T

        task_stage = "PBVS camera aim for destination"
        if not goto(lambda: P_HOME, tol=0.02):
            return False

        target = tray
        start = data.time
        while tray not in tracks and data.time - start < 2.0:
            step_sim()

        if tray not in tracks:
            target = None
            search_direction = BIN_AREA[tray][:2] - camera_position[:2]
            search_direction /= max(float(np.linalg.norm(search_direction)), 1e-8)
            search_position = BIN_AREA[tray].copy()
            search_position[:2] -= 0.25 * search_direction
            search_position[2] = HOVER_Z + 0.10
            task_stage = "PBVS destination camera search"
            if not goto(lambda: search_position, tol=0.02):
                return False
            target = tray
            start = data.time
            while tray not in tracks and data.time - start < 2.0:
                step_sim()

        found_tray = tray in tracks
        if found_tray:
            tray_anchors[tray] = tracks[tray].copy()
            print(
                f"  Pre-pick RGB-D tray reference {tray}: "
                f"{np.round(tracks[tray], 3)}"
            )
        target = None
        R_des = R_HOME.copy()
        task_stage = "PBVS return from destination search"
        returned_home = goto(lambda: P_HOME, tol=0.02)
        if not returned_home:
            return False
        return found_tray

    print(f"\n>>> Picking {cube} -> {tray}")
    task_stage = "Locating destination tray"
    if not ensure_tray_localized():
        return abort(f"camera could not localize {tray} before pickup", True)

    task_stage = "Waiting for target detection"
    target = cube                                     # the camera now looks at this cube
    tracks.pop(cube, None)
    track_last_seen.pop(cube, None)
    sensor.position_filters[cube].reset()
    detection_start = data.time
    while cube not in tracks and data.time - detection_start < 8.0:
        step_sim()
    if cube not in tracks:
        return abort("camera could not reacquire the cube within 8 seconds", True)

    gripper_cmd = GRIPPER_OPEN
    task_stage = "PBVS move to hover"
    if not goto(lambda: np.r_[cube_xy(cube), HOVER_Z], tol=0.012):
        return abort("PBVS hover move failed", True)

    if cube == "Red Cube":                                       # the red cube stops sliding
        red_moving = False
        data.qvel[red_dadr:red_dadr + 6] = 0
    task_stage = "Settling over target"
    wait(0.5)                                                    # let the camera settle on a still cube
    spot = cube_xy(cube).copy()                                  # remember where the cube is

    task_stage = "PBVS descent to grasp"
    if not goto(lambda: np.r_[spot, GRASP_Z], tol=0.008):
        return abort("PBVS descent failed", True)
    cube_body = cube.lower().replace(" ", "_")
    cube_geom_id = model.geom(f"{cube_body}_geom").id
    cube_before_lift = data.xpos[model.body(cube_body).id].copy()
    task_stage = "Closing gripper"
    gripper_cmd = GRIPPER_CLOSED                                 # 3. close the gripper
    wait(0.8)
    tcp_estimate, _ = forward_kinematics(q_est)
    cube_truth = data.xpos[model.body(cube_body).id]
    close_contacts = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if cube_geom_id in (int(contact.geom1), int(contact.geom2)):
            other_geom = int(contact.geom2 if int(contact.geom1) == cube_geom_id else contact.geom1)
            close_contacts.append(model.geom(other_geom).name)
    print(
        "  Grasp evaluation: "
        f"target={np.round(spot, 3)}, TCP={np.round(tcp_estimate, 3)}, "
        f"cube={np.round(cube_truth, 3)}, "
        f"finger contacts={[name for name in close_contacts if 'finger' in name]}"
    )
    tcp_position, _ = forward_kinematics(q_est)
    task_stage = "PBVS lift"
    if not goto(lambda: np.r_[tcp_position[:2], HOVER_Z], tol=0.015):
        return abort("PBVS lift failed; holding state is uncertain")
    cube_after_lift = data.xpos[model.body(cube_body).id].copy()
    lift_height = float(cube_after_lift[2] - cube_before_lift[2])
    holding_cube = lift_height > 0.02
    picked = holding_cube
    if holding_cube:                                             # evaluation from the finger position
        print(
            f"  Cube followed gripper: lift={lift_height:.3f}m "
            f"from {np.round(cube_before_lift, 3)} to {np.round(cube_after_lift, 3)}"
        )
    else:
        print(
            f"  MISSED the cube: lift={lift_height:.3f}m "
            f"from {np.round(cube_before_lift, 3)} to {np.round(cube_after_lift, 3)}"
        )
        gripper_cmd = GRIPPER_OPEN
        return abort("grasp was not retained during lift", True)

    tcp_position, tcp_rotation = forward_kinematics(q_est)
    held_offset = tcp_rotation.T @ (
        cube_after_lift - tcp_position
    )
    print(f"  Held-cube offset in TCP frame: {np.round(held_offset, 3)}")

    def held_shift_xy():
        """World xy shift of the held cube relative to the TCP.

        DIAGNOSTIC: held_offset comes from simulator truth, so this is only
        applied when COMPENSATE_HELD_OFFSET is True."""
        if not COMPENSATE_HELD_OFFSET:
            return np.zeros(2)
        _, rotation = forward_kinematics(q_est)
        return (rotation @ held_offset)[:2]

    def tray_goal(z):
        """Where the TCP should go so the cube ends up over the tray centre."""
        return np.r_[tray_xy(tray) - held_shift_xy(), z]

    carry_failure_detected = False

    def evaluate_carry(stage):
        nonlocal carry_failure_detected
        tcp_position_now, tcp_rotation_now = forward_kinematics(q_est)
        cube_position_now = data.xpos[model.body(cube_body).id].copy()
        offset_now = tcp_rotation_now.T @ (
            cube_position_now - tcp_position_now
        )
        drift = float(np.linalg.norm(offset_now - held_offset))
        finger_contacts = set()
        cube_contact_pairs = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if cube_geom_id in (int(contact.geom1), int(contact.geom2)):
                other_geom = int(
                    contact.geom2
                    if int(contact.geom1) == cube_geom_id
                    else contact.geom1
                )
                name = model.geom(other_geom).name
                cube_contact_pairs.add(name)
                if "finger" in name:
                    finger_contacts.add(name)
        carry_ok = drift <= 0.025 and bool(finger_contacts)
        if not carry_ok:
            carry_failure_detected = True
        if stage != "in-transit carry check" or not carry_ok:
            print(
                f"  Carry check after {stage}: relative-offset drift={drift:.3f}m "
                f"cube={np.round(cube_position_now, 3)} "
                f"TCP={np.round(tcp_position_now, 3)} "
                f"finger contacts={sorted(finger_contacts)} "
                f"cube contact pairs={sorted(cube_contact_pairs)} "
                f"finger_q={data.qpos[FINGER_ADR]:.4f} "
                f"gripper_ctrl={data.ctrl[6]:.4f} "
                f"gripper_force={data.actuator_force[6]:.2f}N"
            )
        return carry_ok

    tcp_position, _ = forward_kinematics(q_est)
    target = None
    task_stage = "PBVS camera-clearance move"
    camera_vantage = tcp_position.copy()
    camera_vantage[2] = max(camera_vantage[2], HOVER_Z + 0.10)
    if not goto(lambda: camera_vantage, tol=0.015, carry_check=evaluate_carry):
        return handle_carry_failure("camera-clearance move failed; cube remains held")
    if not evaluate_carry("camera clearance"):
        return handle_carry_failure("cube slipped during camera-clearance movement")

    task_stage = "Transfer to tray"
    target = tray
    if tray not in tracks:
        return abort(f"no pre-pick RGB-D reference is available for {tray}; cube remains held")
    tray_detection = last_vision_frames["detections"].get(tray)
    tray_raw_position = (
        tray_detection["position_world_raw_m"]
        if tray_detection is not None
        else None
    )
    print(
        f"  RGB-D tray reference {tray}: "
        f"raw={np.round(tray_raw_position, 3) if tray_raw_position is not None else 'occluded'} "
        f"filtered={np.round(tracks[tray], 3)} "
        f"clamped_xy={np.round(tray_xy(tray), 3)} "
        f"age={data.time - track_last_seen.get(tray, data.time):.2f}s"
    )

    task_stage = "PBVS move above tray"
    tray_hover_z = HOVER_Z + 0.10
    if not goto(
        lambda: tray_goal(tray_hover_z),
        tol=0.012,
        carry_check=evaluate_carry,
    ):
        return handle_carry_failure("PBVS move above tray failed; cube remains held")
    if not evaluate_carry("move above tray"):
        return handle_carry_failure("cube slipped while moving above the tray")

    task_stage = "PBVS descend into tray"
    if not goto(
        lambda: tray_goal(DROP_Z),
        tol=0.012,
        carry_check=evaluate_carry,
    ):
        return handle_carry_failure("PBVS tray approach failed; cube remains held")
    if not evaluate_carry("tray approach"):
        return handle_carry_failure("cube slipped before release at the tray")

    task_stage = "Releasing cube"
    wait(RELEASE_SETTLE_S)                                       # let the arm stop swinging

    # Diagnostic: which error is responsible if the cube lands outside?
    cube_now = data.xpos[model.body(cube_body).id]
    tray_true = data.xpos[model.body(TRAY_BODY[tray]).id]
    tcp_now, _ = forward_kinematics(q_est)
    print(
        "  RELEASE DEBUG: "
        f"tracked-true tray err={np.round(tracks[tray][:2] - tray_true[:2], 3)} "
        f"TCP-true tray err={np.round(tcp_now[:2] - tray_true[:2], 3)} "
        f"cube-true tray err={np.round(cube_now[:2] - tray_true[:2], 3)} "
        f"cube z={cube_now[2]:.3f}"
    )

    release_gripper()                                            # ...and let go gradually
    wait(0.5)

    cube_position = data.xpos[model.body(cube_body).id].copy()
    tray_position = data.xpos[model.body(TRAY_BODY[tray]).id].copy()
    in_tray = (
        np.all(np.abs(cube_position[:2] - tray_position[:2]) <= 0.055)
        and 0.0 <= cube_position[2] <= 0.12
    )
    placement_success = bool(holding_cube and in_tray)
    placed = placement_success
    print(
        "  PLACEMENT CHECK: "
        f"{'PASS' if placement_success else 'FAIL'} "
        f"(cube centre {np.round(cube_position, 3)})"
    )
    if not placement_success:
        return abort("cube was released outside the matching tray")

    tcp_position, _ = forward_kinematics(q_est)
    task_stage = "Returning home"
    if not goto(lambda: np.r_[tcp_position[:2], DROP_Z], tol=0.015):
        print("  PBVS retreat after placement did not reach its target")
        return finish(False)
    rotate_base(HOME[0])
    R_des = R_HOME.copy()
    target = None
    for name in CUBES:                                           # cubes were out of view: start their filters fresh
        tracks.pop(name, None)
        track_last_seen.pop(name, None)
        sensor.position_filters[name].reset()
    if not goto(lambda: P_HOME, tol=0.02):
        return abort("PBVS return-home move failed")
    task_stage = "Task complete"
    print(f"<<< {cube} done")
    return finish(True)


def recover_before_retry(cube):
    """Return to a known safe pose and reset vision before one pick retry."""
    global gripper_cmd, target, R_des, task_stage
    print(f"  Recovery: returning to HOME before retrying {cube}")
    task_stage = "Safe recovery to HOME"
    gripper_cmd = GRIPPER_OPEN
    target = None
    R_des = R_HOME.copy()
    if not goto(lambda: P_HOME, tol=0.02):
        print(f"  Recovery failed: could not reach HOME; {cube} will not be retried")
        return False
    rotate_base(HOME[0])
    tracks.pop(cube, None)
    track_last_seen.pop(cube, None)
    sensor.position_filters[cube].reset()
    return True


# --------------------------------------------------
# 5. RUN
# --------------------------------------------------
while True:
    try:
        order = parse_pick_order(
            input("Pick order (e.g. 'r,b,y' or 'yellow red'): "),
            CUBE_TO_TRAY,
        )
        break
    except ValueError as error:
        print(f"Invalid pick order: {error}")
print("Order:", order)

viewer = mujoco.viewer.launch_passive(model, data)    # open the window
if VISION_MONITOR_ENABLED:
    vision_monitor.open()
run_wall_start = time.perf_counter()
for cube in order:
    completed = False
    for attempt in range(1, 3):
        print(f"Attempt {attempt}/2 for {cube}")
        if pick_and_place(cube, CUBE_TO_TRAY[cube]):
            completed = True
            break
        if not last_failure_recoverable or attempt == 2:
            break
        if not recover_before_retry(cube):
            break
    task_results.append(completed)
    if not completed:
        print(
            f"SEQUENCE STOPPED: {cube} was not fully sorted; "
            "later cubes will not be attempted."
        )
        break
run_wall_seconds = time.perf_counter() - run_wall_start
all_requested_completed = len(task_results) == len(order) and all(task_results)
print("\nAll requested cubes sorted." if all_requested_completed else
      "\nSorting run incomplete; review the reported checks.")
joint_tracking_rmse = float("nan")
joint_estimation_rmse = float("nan")
if joint_truth_samples:
    truth_samples = np.asarray(joint_truth_samples)
    reference_samples = np.asarray(joint_reference_samples)
    tracking = tracking_metrics(
        reference_samples,
        truth_samples,
    )
    joint_tracking_rmse = tracking["rmse_rad"]
    estimation = tracking_metrics(
        truth_samples,
        np.asarray(joint_estimate_samples),
    )
    joint_estimation_rmse = estimation["rmse_rad"]
    print(
        "Joint tracking RMSE: "
        f"{tracking['rmse_rad']:.5f} rad "
        f"(max {tracking['max_abs_error_rad']:.5f} rad)"
    )
    print(
        "Joint estimation RMSE: "
        f"{estimation['rmse_rad']:.5f} rad "
        f"(max {estimation['max_abs_error_rad']:.5f} rad)"
    )
    worst_sample, worst_joint = np.unravel_index(
        np.argmax(np.abs(truth_samples - reference_samples)),
        truth_samples.shape,
    )
    print(
        "Largest joint tracking deviation: "
        f"joint {worst_joint + 1} at t={joint_sample_times[worst_sample]:.2f}s "
        f"({truth_samples[worst_sample, worst_joint] - reference_samples[worst_sample, worst_joint]:.4f} rad; "
        f"reference={reference_samples[worst_sample, worst_joint]:.4f}, "
        f"actual={truth_samples[worst_sample, worst_joint]:.4f})"
    )
    away_mask = np.asarray(away_from_home_samples, dtype=bool)
    if np.any(away_mask):
        away_tracking = tracking_metrics(
            reference_samples[away_mask],
            truth_samples[away_mask],
        )
        print(
            "Joint tracking RMSE away from HOME: "
            f"{away_tracking['rmse_rad']:.5f} rad "
            f"(max {away_tracking['max_abs_error_rad']:.5f} rad)"
        )
if camera_position_errors:
    camera_rmse = np.sqrt(np.mean(np.square(camera_position_errors)))
    print(f"Estimated camera-position RMSE: {camera_rmse:.5f} m")
else:
    camera_rmse = float("nan")
localization_errors = [
    error
    for errors in object_localization_errors.values()
    for error in errors
]
if localization_errors:
    localization_rmse = np.sqrt(np.mean(np.square(localization_errors)))
    print(f"RGB-D object-localization RMSE: {localization_rmse:.5f} m")
    for name, errors in object_localization_errors.items():
        if errors:
            per_object_rmse = np.sqrt(np.mean(np.square(errors)))
            print(
                f"  {name} localization RMSE: {per_object_rmse:.5f}m "
                f"(max {max(errors):.5f}m, {len(errors)} detections, "
                f"{localization_outlier_counts[name]} errors > 0.05m)"
            )
    rejected = {
        name: count for name, count in rejected_detection_counts.items() if count
    }
    if rejected:
        print(f"Rejected out-of-workspace colour detections: {rejected}")
else:
    localization_rmse = float("nan")
if pbvs_position_errors:
    pbvs_rmse = float(np.sqrt(np.mean(np.square(pbvs_position_errors))))
    print(
        "PBVS reached-position error: "
        f"RMSE={pbvs_rmse:.5f} m "
        f"(max {max(pbvs_position_errors):.5f} m; "
        f"{len(pbvs_position_errors)} reached waypoints)"
    )
else:
    pbvs_rmse = float("nan")
model_residual_rmse = float("nan")
if model_prediction_residuals:
    residuals = np.asarray(model_prediction_residuals)
    model_residual_rmse = float(np.sqrt(np.mean(residuals**2)))
    away_mask = np.asarray(away_from_home_samples, dtype=bool)
    if np.any(away_mask):
        print(
            "Position-actuator model one-step residual: "
            f"{np.sqrt(np.mean(residuals**2)):.3e} overall; "
            f"{np.sqrt(np.mean(residuals[away_mask]**2)):.3e} away from HOME"
        )
    else:
        print(
            "Position-actuator model one-step residual: "
            f"{np.sqrt(np.mean(residuals**2)):.3e} overall"
        )
if VISION_MONITOR_ENABLED:
    print(
        "Vision performance: "
        f"capture={1000 * np.mean(vision_capture_times):.1f}ms/frame, "
        f"detection={1000 * np.mean(vision_detection_times):.1f}ms/frame, "
        f"display={1000 * np.mean(vision_display_times):.1f}ms/frame, "
        f"monitor FPS={vision_monitor.fps:.1f}"
        if vision_capture_times and vision_detection_times and vision_display_times
        else "Vision performance: no monitor frames were displayed"
    )
else:
    print("Vision monitor disabled (set VISION_MONITOR=1 to enable).")
if run_wall_seconds > 0:
    print(
        f"Simulation runtime: {data.time:.2f}s simulated / "
        f"{run_wall_seconds:.2f}s wall "
        f"({data.time / run_wall_seconds:.2f}x real time)"
    )
if np.any(actuator_saturation_counts):
    print(
        "LQR actuator saturation counts by joint: "
        f"{actuator_saturation_counts.tolist()}"
    )
if placement_results:
    print(
        f"Placement success rate: {sum(placement_results)}/{len(placement_results)} attempts "
        f"({100 * np.mean(placement_results):.1f}%)"
    )
if pick_results:
    print(
        f"Pick success rate: {sum(pick_results)}/{len(pick_results)} attempts "
        f"({100 * np.mean(pick_results):.1f}%)"
    )
retries_used = max(0, len(pick_results) - len(task_results))
print(
    f"Requested cubes fully completed: {sum(task_results)}/{len(order)}; "
    f"retries used: {retries_used}"
)
print(
    f"Complete requested sequence: {'PASS' if all_requested_completed else 'FAIL'}"
)
print(
    f"Robustness conditions: cube placement seed={random_seed if random_seed is not None else 'random'}; "
    f"encoder noise "
    f"std={ENCODER_NOISE_STD:.4f} rad; red-cube motion amplitude "
    f"{RED_AMPLITUDE:.3f} m"
)

results_path = Path(__file__).resolve().with_name("trial_results.csv")
result_columns = [
    "timestamp", "seed", "requested_order", "sequence_success",
    "requested_cubes", "completed_cubes", "attempts", "retries",
    "pick_successes", "placement_successes", "joint_tracking_rmse_rad",
    "joint_estimation_rmse_rad", "pbvs_position_rmse_m",
    "localization_rmse_m", "camera_position_rmse_m",
    "actuator_model_residual_rmse", "encoder_noise_std_rad",
]
write_header = not results_path.exists()
with results_path.open("a", newline="", encoding="utf-8") as results_file:
    writer = csv.DictWriter(results_file, fieldnames=result_columns)
    if write_header:
        writer.writeheader()
    writer.writerow({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": random_seed if random_seed is not None else "random",
        "requested_order": ",".join(cube.removesuffix(" Cube").lower() for cube in order),
        "sequence_success": all_requested_completed,
        "requested_cubes": len(order),
        "completed_cubes": sum(task_results),
        "attempts": len(pick_results),
        "retries": retries_used,
        "pick_successes": sum(pick_results),
        "placement_successes": sum(placement_results),
        "joint_tracking_rmse_rad": joint_tracking_rmse,
        "joint_estimation_rmse_rad": joint_estimation_rmse,
        "pbvs_position_rmse_m": pbvs_rmse,
        "localization_rmse_m": localization_rmse,
        "camera_position_rmse_m": camera_rmse,
        "actuator_model_residual_rmse": model_residual_rmse,
        "encoder_noise_std_rad": ENCODER_NOISE_STD,
    })
print(f"Trial results appended to {results_path.name}")

renderer.close()
if VISION_MONITOR_ENABLED:
    vision_monitor.close()
viewer.close()
