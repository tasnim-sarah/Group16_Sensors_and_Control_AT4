
import re
import sys
import time
from pathlib import Path
 
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from camera_module import CameraSensor, aim_camera_once
 
# Scene names. If your XML uses different names, change them here.
COLOURS = {
    'r': dict(name='red', cube='red_cube', bin='red_bin', drop='red_bin_drop'),
    'b': dict(name='blue', cube='blue_cube', bin='blue_bin', drop='blue_bin_drop'),
    'y': dict(name='yellow', cube='yellow_cube', bin='yellow_bin',
              drop='yellow_bin_drop'),
}
 
 
def parse_order(text):
    """'r b y', 'rby', 'red, blue' ... -> ['r', 'b', 'y'] (None if invalid)."""
    words = {'r': 'r', 'red': 'r', 'b': 'b', 'blue': 'b', 'y': 'y', 'yellow': 'y'}
    order = []
    for token in re.findall(r'[a-z]+', text.lower()):
        if token in words:
            order.append(words[token])
        elif set(token) <= set('rby'):
            order.extend(token)
        else:
            return None
    if not order or len(set(order)) != len(order):
        return None
    return order
 
 
def ask_order():
    if len(sys.argv) > 1:
        order = parse_order(' '.join(sys.argv[1:]))
        if order:
            return order
        print('Could not read a pick order from the command line.')
    while True:
        try:
            text = input('Pick order (r = red, b = blue, y = yellow), e.g. "r b y": ')
        except EOFError:
            raise SystemExit('No pick order given.')
        order = parse_order(text)
        if order:
            return order
        print('  Use r, b and/or y, each at most once (for example: r b y).')
 
 
ORDER = ask_order()
print('Pick order:', ' -> '.join(COLOURS[k]['name'] for k in ORDER))
 
HERE = Path(__file__).resolve().parent
spec = mujoco.MjSpec.from_file(str(HERE / 'universal_robots_ur3e' / 'pick_scene.xml'))
for key in list(spec.keys):
    spec.delete(key)
 
wrist = spec.body('wrist_3_link')
wrist.add_geom(name='camera_shell', type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[.045, .0125, .0125], pos=[0, .08, .05],
               rgba=[.15, .15, .15, 1], mass=.072, contype=0, conaffinity=0)
cam = wrist.add_camera(name='wrist_cam', pos=[.0325, .095, .05], fovy=51.28)
cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]
 
g = wrist.add_body(name='gripper_base', pos=[0, .09215, 0])
g.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
g.alt.xyaxes = [0, 0, 1, 1, 0, 0]
g.add_geom(name='palm', type=mujoco.mjtGeom.mjGEOM_BOX,
           size=[.02, .05, .01], pos=[0, 0, .01], mass=.25,
           rgba=[.2, .2, .2, 1], contype=1, conaffinity=1, group=0)
 
FINGER_TRAVEL = .030     # per-finger stroke (m)
FINGER_OFFSET = .040     # finger centre distance from gripper axis when open (m)
KP_GRIP = 400.0          # N/m per finger
SQUEEZE_FORCE = 5.0      # N per finger once contact is confirmed
MAX_GRIP_FORCE = 30.0
 
for side, sign in [('left', 1), ('right', -1)]:
    f = g.add_body(name=f'finger_{side}', pos=[0, sign * FINGER_OFFSET, .04])
    f.add_joint(name=f'finger_{side}', type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, -sign, 0], range=[0, FINGER_TRAVEL],
                damping=5, armature=.001)
    f.add_geom(name=f'finger_{side}_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[.010, .006, .020], mass=.05, rgba=[.3, .3, .3, 1],
               friction=[3.0, .01, .001], condim=4,
               contype=1, conaffinity=1, group=0)
g.add_site(name='tcp', pos=[0, 0, .045], size=[.006, 0, 0], rgba=[1, 0, 1, 1])
 
# One actuator per finger: with equal targets the cube self-centres between
# the pads and both pads push equally.
for side in ('left', 'right'):
    act = spec.add_actuator(name=f'gripper_{side}', target=f'finger_{side}',
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=KP_GRIP)
    act.ctrlrange = [0, FINGER_TRAVEL]
    act.forcerange = [-MAX_GRIP_FORCE, MAX_GRIP_FORCE]
 
model = spec.compile()
# Stiffer friction handling for grasping (reduces creep/slip at the pads).
model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
model.opt.impratio = 10
 
TCP = model.site('tcp').id
CAM = model.camera('wrist_cam').id
joint_ids = [model.joint(n).id for n in
             ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
qadr = [int(model.jnt_qposadr[j]) for j in joint_ids]
dadr = [int(model.jnt_dofadr[j]) for j in joint_ids]
q_min = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -2*np.pi
                  for j in joint_ids])
q_max = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else 2*np.pi
                  for j in joint_ids])
 
# ---- Arm sag fix --------------------------------------------------------
ARM_GRAVITY_COMP = True  # Keep enabled: compensates arm/gripper weight.
if ARM_GRAVITY_COMP:
    arm_root = int(model.jnt_bodyid[joint_ids[0]])   # shoulder_link
    comp_count = 0
    for b in range(1, model.nbody):
        k = b
        while k > 0 and k != arm_root:
            k = int(model.body_parentid[k])
        if k == arm_root:                 # arm, wrist, camera, gripper, fingers
            model.body_gravcomp[b] = 1.0
            comp_count += 1
    print(f'Gravity compensation on for {comp_count} arm/gripper bodies.')
 
data = mujoco.MjData(model)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
rng = np.random.default_rng()
bands = [(0.28, .42, .05, .20), (.28, .42, -.08, .05),
         (.28, .42, -.20, -.08)]
rng.shuffle(bands)
for name, (x0, x1, y0, y1) in zip(
        ['red_cube_joint', 'yellow_cube_joint', 'blue_cube_joint'], bands):
    adr = int(model.jnt_qposadr[model.joint(name).id])
    data.qpos[adr:adr+2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
GRIP_IDS = [model.actuator('gripper_left').id, model.actuator('gripper_right').id]
 
 
def set_grip(target):
    data.ctrl[GRIP_IDS] = target
 
 
set_grip(0.)
mujoco.mj_forward(model, data)
 
print('Camera alignment:', round(aim_camera_once(model, data), 5))
renderer = mujoco.Renderer(model, width=640, height=480)
try:
    sensor = CameraSensor(model, renderer)
except Exception:
    renderer.close()
    raise
 
 
def detect_all(show=False):
    """RGB-D detection of every cube from the current (home) camera pose."""
    rgb, depth = sensor.capture(data)
    found = sensor.detect(data, rgb, depth)
    if show:
        annotated = rgb.copy()
        for label, item in found.items():
            u, v = item['pixel']
            cv2.drawContours(annotated, [item['contour']], -1, (0, 255, 0), 2)
            cv2.putText(annotated, label, (max(0, int(u)-50), max(20, int(v)-12)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.imshow('RGB-D detection (press any key)',
                   cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return found
 
 
try:
    detections = detect_all(show=True)
except Exception:
    renderer.close()
    raise
print('Detected:', list(detections))
 
 
def find_detection(found, colour):
    for label, item in found.items():
        if label.lower().startswith(colour['name']):
            return item
    return None
 
 
CUBE_CENTER_Z = 0.021  # 40 mm cube resting on tabletop Z=0, ~1 mm clearance
# Finger pad centres are at gripper local Z=.040, TCP at Z=.045. With +Z
# pointing down, the pad centre sits 5 mm ABOVE the TCP in world.
GRASP_Z_OFFSET = +.007  # TCP Z = 0.028 m; allows pad clearance while gripping cube sides.
HOVER_ABOVE_CUBE = .20
TRANSFER_HEIGHT = .24  # Clearance above bin walls, not collision avoidance
BIN_RELEASE_ABOVE_SITE = .02  # TCP height above the bin drop site at release
GRASP_POSITION_TOL = .0035
GRASP_ORIENTATION_TOL = .07
ALIGN_TO_CUBE_AXES = True  # snap gripper closing axis to world X or Y
MIN_CAMERA_Z = .025   # Safety check on camera position only; NOT full collision detection.
 
# Live wrist-camera monitor (uses the actual MuJoCo camera attached to wrist_3_link).
LIVE_WINDOW = 'LIVE wrist_cam  |  press Q to stop camera and simulation'
CAMERA_FPS = 12
last_camera_frame = 0.0
camera_window_open = False
stop_requested = False
 
 
def update_live_camera(force=False):
    global last_camera_frame, camera_window_open, stop_requested
    if stop_requested:
        return False
    now = time.monotonic()
    if not force and now - last_camera_frame < 1.0 / CAMERA_FPS:
        return True
    if camera_window_open:
        try:
            if cv2.getWindowProperty(LIVE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                camera_window_open = False
                return True
        except cv2.error:
            camera_window_open = False
            return True
    else:
        if last_camera_frame != 0:
            return True
        cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
        camera_window_open = True
 
    renderer.update_scene(data, camera='wrist_cam')
    rgb = renderer.render()
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, 'LIVE wrist camera', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow(LIVE_WINDOW, frame)
    last_camera_frame = now
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_requested = True
        return False
    return True
 
 
def cleanup_camera():
    renderer.close()
    cv2.destroyAllWindows()
 
 
def step_hold(viewer, duration):
    for _ in range(max(1, int(duration / model.opt.timestep))):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
        viewer.sync()
        if not update_live_camera():
            return False
    return True
 
 
class PickFailed(Exception):
    pass
 
 
def need(condition, message):
    if not condition:
        raise PickFailed(message)
 
 
def down_orientation(rot90=False, verbose=False):
    """Target TCP rotation with local +Z pointing vertically down.
 
    The horizontal heading of the TCP X axis is kept and snapped to the nearest
    world axis so the closing axis (TCP Y) is parallel to the cube faces.
    rot90=True turns the gripper a further 90 deg about vertical, which swaps
    the closing axis between world X and world Y.
    """
    mujoco.mj_forward(model, data)
    old = data.site_xmat[TCP].reshape(3, 3).copy()
    x = old[:, 0].copy()
    x[2] = 0.
    if np.linalg.norm(x) < 1e-5:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    if verbose:
        print(f'Gripper X heading: {np.degrees(np.arctan2(x[1], x[0])):.1f} deg')
    if ALIGN_TO_CUBE_AXES:
        if abs(x[0]) >= abs(x[1]):
            x = np.array([np.sign(x[0]), 0., 0.])
        else:
            x = np.array([0., np.sign(x[1]), 0.])
    if rot90:
        x = np.array([-x[1], x[0], 0.])
    z = np.array([0., 0., -1.])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack((x, y, z))
 
 
def rotation_error(current, desired):
    """World-frame orientation error vector for a geometric angular Jacobian."""
    return 0.5 * (np.cross(current[:, 0], desired[:, 0])
                  + np.cross(current[:, 1], desired[:, 1])
                  + np.cross(current[:, 2], desired[:, 2]))
 
 
def servo_tick(ref, v_ref, R_target, q_cmd, dt, kp_pos=4.0, kp_rot=1.5):
    """One damped-Jacobian control tick tracking a moving reference point.
 
    v_ref is the reference velocity (feed-forward) so the arm does not lag
    behind a moving target. q_cmd persists between ticks (no reset to the
    measured joints), so the arm never brakes between points.
    """
    mujoco.mj_forward(model, data)
    pos_err = ref - data.site_xpos[TCP]
    ori_err = rotation_error(data.site_xmat[TCP].reshape(3, 3), R_target)
    jpos = np.zeros((3, model.nv))
    jrot = np.zeros((3, model.nv)) 
    mujoco.mj_jacSite(model, data, jpos, jrot, TCP)
    J = np.vstack((jpos[:, dadr], jrot[:, dadr]))
    command = np.concatenate((v_ref + kp_pos * pos_err, kp_rot * ori_err))
    dq = J.T @ np.linalg.solve(J @ J.T + .085**2 * np.eye(6), command)
    dq = np.clip(dq, -.6, .6)
    measured = np.asarray(data.qpos[qadr])
    q_cmd = np.clip(q_cmd, measured - .12, measured + .12)
    q_cmd = np.clip(q_cmd + dq * dt, q_min, q_max)
    data.ctrl[:6] = q_cmd
    return q_cmd, float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err))
 
 
def hold_tcp_pose(viewer, duration, xyz, R_target):
    """Actively hold the TCP pose during finger closing/contact settling."""
    dt = .02
    n = max(1, int(np.ceil(duration / dt)))
    q_cmd = np.array(data.ctrl[:6], dtype=float)
    for _ in range(n):
        q_cmd, _, _ = servo_tick(np.asarray(xyz, dtype=float), np.zeros(3),
                                 R_target, q_cmd, dt, kp_pos=5.0, kp_rot=2.0)
        if not step_hold(viewer, dt):
            return False
    return True
 
 
def move_smooth(viewer, target_xyz, R_target, label, avg_speed=.10,
                min_time=.8, settle_tol=.005, settle_time=3.0):
    """Continuous straight-line move with a minimum-jerk speed profile.
 
    One smooth motion (no stop-start between waypoints). Also turns the tool
    to R_target and only reports success once position AND orientation agree.
    """
    target_xyz = np.asarray(target_xyz, dtype=float)
    mujoco.mj_forward(model, data)
    start = data.site_xpos[TCP].copy()
    R_now = data.site_xmat[TCP].reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(R_now.T @ R_target) - 1) / 2, -1, 1)))
    dist = float(np.linalg.norm(target_xyz - start))
    dt = max(model.opt.timestep, .02)
    n = max(1, int(round(max(min_time, dist / avg_speed, angle / .8) / dt)))
    print(f'{label}: {np.round(start, 3)} -> {np.round(target_xyz, 3)} '
          f'({dist:.3f} m, {np.degrees(angle):.0f} deg, {n * dt:.1f} s)')
    q_cmd = np.array(data.ctrl[:6])
    prev = start.copy()
    for k in range(1, n + 1):
        s = k / n
        ref = start + (target_xyz - start) * (10*s**3 - 15*s**4 + 6*s**5)
        q_cmd, _, _ = servo_tick(ref, (ref - prev) / dt, R_target, q_cmd, dt)
        prev = ref
        if data.cam_xpos[CAM, 2] < MIN_CAMERA_Z:
            print('  STOP: camera centre is too close to/below tabletop.')
            return False
        if not step_hold(viewer, dt):
            return False
    err = ori = float('inf')
    for _ in range(int(settle_time / dt)):
        q_cmd, err, ori = servo_tick(target_xyz, np.zeros(3), R_target, q_cmd, dt)
        if err < settle_tol and ori < GRASP_ORIENTATION_TOL:
            print(f'  Reached: XYZ error {err:.4f} m, orientation error {ori:.3f}')
            return True
        if not step_hold(viewer, dt):
            return False
    print(f'  FAILED to settle: XYZ error {err:.4f} m, orientation error {ori:.3f}')
    return False
 
 
def go_home(viewer):
    """Smooth joint-space return to the start pose (swings round the base)."""
    q0 = np.array(data.qpos[qadr])
    duration = max(2.0, float(np.max(np.abs(HOME - q0))) / .9)
    dt = .02
    n = int(round(duration / dt))
    print(f'Returning to start pose ({duration:.1f} s)')
    for k in range(1, n + 1):
        s = k / n
        data.ctrl[:6] = q0 + (HOME - q0) * (10*s**3 - 15*s**4 + 6*s**5)
        if not step_hold(viewer, dt):
            return False
    return step_hold(viewer, .8)
 
 
# ---- Per-cube state (ground truth used ONLY for validation printouts) ----
CUBE_BODY = CUBE_GEOM = BIN_DROP = BIN_BODY = None
INITIAL_CUBE_Z = 0.0
 
 
def lookup(kind, name):
    try:
        return int(getattr(model, kind)(name).id)
    except Exception:
        raise SystemExit(f"The scene has no {kind} called '{name}'. "
                         f"Check pick_scene.xml and the COLOURS table at the top.")
 
 
def select_cube(colour):
    global CUBE_BODY, CUBE_GEOM, BIN_DROP, BIN_BODY, INITIAL_CUBE_Z
    CUBE_BODY = lookup('body', colour['cube'])
    CUBE_GEOM = int(model.body_geomadr[CUBE_BODY])
    BIN_DROP = lookup('site', colour['drop'])
    BIN_BODY = lookup('body', colour['bin'])
    mujoco.mj_forward(model, data)
    INITIAL_CUBE_Z = float(data.xpos[CUBE_BODY, 2])
    print(f'Cube mass={model.body_mass[CUBE_BODY]:.3f} kg, '
          f'friction={np.round(model.geom_friction[CUBE_GEOM], 3)}, '
          f'size={np.round(model.geom_size[CUBE_GEOM], 4)}')
 
 
def carried_cube_state():
    mujoco.mj_forward(model, data)
    cube_pos = data.xpos[CUBE_BODY].copy()
    tcp_pos = data.site_xpos[TCP].copy()
    return cube_pos, float(np.linalg.norm(cube_pos[:2] - tcp_pos[:2]))
 
 
def finger_contacts():
    """Count cube-pad contacts reported by MuJoCo contact solver."""
    pad_ids = {model.geom('finger_left_pad').id, model.geom('finger_right_pad').id}
    touching = set()
    for c in data.contact:
        ids = {int(c.geom1), int(c.geom2)}
        if CUBE_GEOM in ids:
            touching.update(ids & pad_ids)
    return len(touching)
 
 
def geom_label(gid):
    name = model.geom(gid).name
    if name:
        return name
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[gid]))
    return f'{body}:geom{gid}'
 
 
def print_pad_contacts():
    """List EVERYTHING touching a finger pad (cube, table, other objects)."""
    pad_ids = {model.geom('finger_left_pad').id, model.geom('finger_right_pad').id}
    seen = False
    for c in data.contact:
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in pad_ids or g2 in pad_ids:
            seen = True
            print(f'    pad contact: {geom_label(g1)} <-> {geom_label(g2)}, '
                  f'dist={float(c.dist):.4f} m')
    if not seen:
        print('    pad contact: none')
 
 
def print_gripper_diagnostics(stage):
    mujoco.mj_forward(model, data)
    left = data.geom_xpos[model.geom('finger_left_pad').id].copy()
    right = data.geom_xpos[model.geom('finger_right_pad').id].copy()
    tcp = data.site_xpos[TCP].copy()
    lp = float(data.qpos[model.jnt_qposadr[model.joint('finger_left').id]])
    rp = float(data.qpos[model.jnt_qposadr[model.joint('finger_right').id]])
    pad_mid = (left + right) / 2
    pad_sep = np.linalg.norm(left - right)
    inner_gap = pad_sep - .012  # pad half-thickness 0.006 m each
    print(f'  {stage}: TCP={np.round(tcp, 4)} pad midpoint={np.round(pad_mid, 4)}')
    print(f'  pad centre separation={pad_sep:.4f} m, '
          f'est. inner gap={inner_gap:.4f} m (cube width .040 m)')
    print(f'  finger slider positions: left={lp:.4f} m, right={rp:.4f} m; '
          f'pad contacts={finger_contacts()}/2')
    closing_axis = data.site_xmat[TCP].reshape(3, 3)[:, 1]
    cube_offset = data.xpos[CUBE_BODY] - pad_mid
    print(f'  TRUE cube - pad midpoint = {np.round(cube_offset, 4)} m; '
          f'along closing axis = {float(np.dot(cube_offset, closing_axis))*1000:.1f} mm')
    print_pad_contacts()
 
 
def close_until_cube_contact(viewer, hold_xyz, hold_R):
    """Ramp both finger targets until BOTH pads touch the cube, then squeeze."""
    l_adr = int(model.jnt_qposadr[model.joint('finger_left').id])
    r_adr = int(model.jnt_qposadr[model.joint('finger_right').id])
    target = 0.0
    both_contact_streak = 0
    print('Closing gradually; stopping on two-sided pad contact.')
    while target < FINGER_TRAVEL - 1e-9:
        target = min(target + .0005, FINGER_TRAVEL)
        set_grip(target)
        if not hold_tcp_pose(viewer, .08, hold_xyz, hold_R):
            return False
        contacts = finger_contacts()
        both_contact_streak = both_contact_streak + 1 if contacts == 2 else 0
        if contacts:
            print(f'  target={target:.4f} m; pads touching={contacts}/2; '
                  f'measured joints=({data.qpos[l_adr]:.4f}, '
                  f'{data.qpos[r_adr]:.4f}) m')
        if both_contact_streak >= 2:
            q_mean = .5 * (float(data.qpos[l_adr]) + float(data.qpos[r_adr]))
            hold_target = min(q_mean + SQUEEZE_FORCE / KP_GRIP, FINGER_TRAVEL)
            set_grip(hold_target)
            print(f'  BOTH PAD CONTACTS at {q_mean:.4f} m: clamping with '
                  f'target {hold_target:.4f} m (~{SQUEEZE_FORCE:.1f} N per finger).')
            if not hold_tcp_pose(viewer, .40, hold_xyz, hold_R):
                return False
            print_gripper_diagnostics('After clamp')
            if finger_contacts() == 2:
                return True
            print('  Contact disappeared during settling; aborting before lift.')
            return False
    print('  STOP: fingers reached full travel without two-sided contact.')
    print_gripper_diagnostics('At full travel')
    return False
 
 
def check_carry(stage, min_rise=.020, max_xy_offset=.028, max_rel_z=.040):
    cube_xyz, xy_error = carried_cube_state()
    mujoco.mj_forward(model, data)
    tcp_z = float(data.site_xpos[TCP, 2])
    relative_z = float(cube_xyz[2] - tcp_z)
    rise = float(cube_xyz[2] - INITIAL_CUBE_Z)
    contacts = finger_contacts()
    print(f'  {stage}: cube XYZ={np.round(cube_xyz, 3)}, '
          f'XY offset={xy_error:.3f} m, cube rise={rise:.3f} m, '
          f'cube-TCP Z={relative_z:.3f} m, pad contacts={contacts}/2')
    if rise < min_rise or xy_error > max_xy_offset or abs(relative_z) > max_rel_z:
        print('  STOP: measured cube displacement indicates an insecure carry.')
        return False
    if contacts < 2:
        print('  STOP: both pads must still contact the cube during carry.')
        return False
    return True
 
 
def in_pickup_area(c):
    return .24 <= c[0] <= .46 and -.23 <= c[1] <= .23 and -.01 <= c[2] <= .07
 
 
def choose_rot90(target_xy, others_xy):
    """Closing axis is world X by default. Switch to Y if another cube sits in
    the swept area of the open jaws (jaws reach ~46 mm each side of centre)."""
    def conflicts(axis):
        count = 0
        for ox, oy in others_xy:
            dx, dy = abs(ox - target_xy[0]), abs(oy - target_xy[1])
            along, across = (dx, dy) if axis == 'x' else (dy, dx)
            if along < .076 and across < .040:
                count += 1
        return count
    return conflicts('y') < conflicts('x')
 
 
def pick_and_place(viewer, colour, detection, others_xy):
    """Pick one cube from its RGB-D detection and drop it in its own bin.
 
    Raises PickFailed if any safety check fails. Returns True if the cube
    ended up inside the bin (ground-truth check, evaluation only).
    """
    select_cube(colour)
    print(f'\n=== {colour["name"].upper()} cube -> {colour["name"]} bin ===')
    visible_xyz = np.asarray(detection['position_world_m'], dtype=float)
    center = visible_xyz.copy()
    center[2] = CUBE_CENTER_Z
    print('Surface XYZ (RGB-D):', np.round(visible_xyz, 4))
    print('Approx. grasp center XYZ:', np.round(center, 4))
    need(in_pickup_area(center), 'RGB-D target outside the pickup area')
 
    rot90 = choose_rot90(center[:2], others_xy)
    if rot90:
        print('Another cube is close along the jaw axis: closing along world Y.')
    target_R = down_orientation(rot90, verbose=True)
    grasp = center.copy()
    grasp[2] += GRASP_Z_OFFSET
    hover = grasp.copy()
    hover[2] += HOVER_ABOVE_CUBE
 
    set_grip(0.)
    mujoco.mj_forward(model, data)
    start_xyz = data.site_xpos[TCP].copy()
    need(move_smooth(viewer, start_xyz, target_R, 'Point gripper vertically down',
                     min_time=.8), 'cannot align gripper')
    need(move_smooth(viewer, hover, target_R, 'Hover over cube', avg_speed=.25,
                     min_time=1.0, settle_tol=.007), 'hover unreachable')
    need(move_smooth(viewer, grasp, target_R, 'Descend', avg_speed=.06,
                     min_time=1.5, settle_tol=GRASP_POSITION_TOL),
         'could not descend accurately')
 
    mujoco.mj_forward(model, data)
    for pad in ('finger_left_pad', 'finger_right_pad'):
        gid = model.geom(pad).id
        Rpad = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        z_extent = float(np.sum(np.abs(Rpad[2, :]) * half))
        lowest = float(data.geom_xpos[gid, 2] - z_extent)
        print(f'  {pad}: lowest point Z={lowest:.4f} m')
        need(lowest >= .006, 'finger pad too close to / below tabletop')
 
    print_gripper_diagnostics('Before closing')
    grasp_hold_xyz = data.site_xpos[TCP].copy()
    print('Holding TCP at', np.round(grasp_hold_xyz, 4), 'while closing')
    need(close_until_cube_contact(viewer, grasp_hold_xyz, target_R),
         'stable two-sided contact was not confirmed (no lift attempted)')
 
    # Gentle slip test, then clear the table. Smooth continuous motions.
    lift_xy = grasp_hold_xyz[:2].copy()
    test_lift_z = float(grasp_hold_xyz[2]) + .065
    need(move_smooth(viewer, [lift_xy[0], lift_xy[1], test_lift_z], target_R,
                     'Slow initial lift', avg_speed=.04, min_time=1.5),
         'initial lift failed')
    need(step_hold(viewer, .35), 'viewer closed')
    need(check_carry('After initial lift', min_rise=.020),
         'cube slipping during lift')
    need(move_smooth(viewer, [lift_xy[0], lift_xy[1], TRANSFER_HEIGHT], target_R,
                     'Clear table', avg_speed=.10), 'cannot reach transport height')
    need(check_carry('Before transfer', min_rise=.065), 'cube lost before transfer')
 
    mujoco.mj_forward(model, data)
    bin_xyz = data.site_xpos[BIN_DROP].copy()
    print(f'{colour["name"].capitalize()} bin drop site (known fixture):',
          np.round(bin_xyz, 3))
    above_bin = np.array([bin_xyz[0], bin_xyz[1], TRANSFER_HEIGHT])
    release_z = float(bin_xyz[2]) + BIN_RELEASE_ABOVE_SITE
    need(move_smooth(viewer, above_bin, target_R, 'Transfer above bin',
                     avg_speed=.18, min_time=2.0), 'bin transfer pose unreachable')
    need(check_carry('Over bin', min_rise=.065), 'cube lost during transfer')
    need(move_smooth(viewer, [above_bin[0], above_bin[1], release_z], target_R,
                     'Lower over bin', avg_speed=.08, min_time=1.0),
         'release height unreachable')
    need(check_carry('Before release', min_rise=.040), 'cube lost before release')
 
    print('Opening gripper above bin...')
    set_grip(0.0)
    need(step_hold(viewer, 1.5), 'viewer closed')
    if not move_smooth(viewer, [above_bin[0], above_bin[1], TRANSFER_HEIGHT],
                       target_R, 'Retreat', avg_speed=.12):
        print('Warning: released cube but retreat did not finish.')
    need(step_hold(viewer, .8), 'viewer closed')
 
    mujoco.mj_forward(model, data)
    final_cube = data.xpos[CUBE_BODY].copy()
    final_bin = data.xpos[BIN_BODY].copy()
    inside_xy = bool(np.all(np.abs(final_cube[:2] - final_bin[:2]) < .045))
    inside_z = bool(.005 < final_cube[2] < .075)
    placed = inside_xy and inside_z
    print(f'Final {colour["name"]} cube XYZ:', np.round(final_cube, 3),
          '| bin XYZ:', np.round(final_bin, 3))
    print(f'PLACEMENT CHECK ({colour["name"]}):',
          'PASS' if placed else 'NOT CONFIRMED: cube may have missed bin')
    return placed
 
 
results = {}
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if not step_hold(viewer, 1.0):
            raise SystemExit('Viewer closed.')
 
        for index, key in enumerate(ORDER):
            colour = COLOURS[key]
            item = find_detection(detections, colour)
            if item is None:
                print(f'\n{colour["name"].capitalize()} cube not detected '
                      f'(saw {list(detections)}): stopping.')
                break
            others_xy = []
            for other_key, other in COLOURS.items():
                if other_key != key and other_key not in results:
                    other_item = find_detection(detections, other)
                    if other_item is not None:
                        others_xy.append(
                            np.asarray(other_item['position_world_m'], float)[:2])
            try:
                results[key] = pick_and_place(viewer, colour, item, others_xy)
            except PickFailed as exc:
                if not viewer.is_running() or stop_requested:
                    print('\nViewer closed.')
                else:
                    print(f'\nSTOPPED on {colour["name"]} cube: {exc}')
                break
            if index < len(ORDER) - 1:
                if not go_home(viewer):
                    print('\nViewer closed.')
                    break
                detections = detect_all()
                print('Detected:', list(detections))
 
        print('\n=== SUMMARY ===')
        for key in ORDER:
            state = results.get(key)
            text = ('placed in bin' if state else
                    'put down but bin not confirmed' if state is False else
                    'not done')
            print(f'  {COLOURS[key]["name"]:>6}: {text}')
 
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if not update_live_camera():
                break
            time.sleep(model.opt.timestep)
finally:
    cleanup_camera()