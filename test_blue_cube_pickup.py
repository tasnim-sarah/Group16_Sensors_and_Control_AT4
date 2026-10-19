"""RGB-D yellow-cube pickup and yellow-bin delivery experiment.

Separate from the team's main scene; does not change camera_module.py/XML.
Requires MuJoCo, OpenCV, NumPy and camera_module.py.

IMPORTANT: Simulation-only experimental controller, not collision avoidance.
It checks rise, relative pose and finger contacts; aborts if the grasp slips.
Bin site poses are used as known destination coordinates, not visual detections.
"""
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from camera_module import CameraSensor, aim_camera_once

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
           size=[.02, .045, .01], pos=[0, 0, .01], mass=.25,
           rgba=[.2, .2, .2, 1], contype=1, conaffinity=1, group=0)
FINGER_TRAVEL = .022
for side, sign in [('left', 1), ('right', -1)]:
    f = g.add_body(name=f'finger_{side}', pos=[0, sign * .032, .04])
    f.add_joint(name=f'finger_{side}', type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, -sign, 0], range=[0, FINGER_TRAVEL],
                damping=5, armature=.001)
    f.add_geom(name=f'finger_{side}_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[.010, .006, .020], mass=.05, rgba=[.3, .3, .3, 1],
               friction=[3.0, .01, .001], condim=4,
               contype=1, conaffinity=1, group=0)
g.add_site(name='tcp', pos=[0, 0, .045], size=[.006, 0, 0], rgba=[1, 0, 1, 1])
spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT,
                  objtype=mujoco.mjtObj.mjOBJ_JOINT,
                  name1='finger_right', name2='finger_left')
a = spec.add_actuator(name='gripper', target='finger_left',
                      trntype=mujoco.mjtTrn.mjTRN_JOINT)
a.set_to_position(kp=400)
a.ctrlrange = [0, FINGER_TRAVEL]
a.forcerange = [-30, 30]  # Simulation-only pad actuation; test contact

model = spec.compile()
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
GRIP_CTRL = model.actuator('gripper').id
data.ctrl[GRIP_CTRL] = 0.
mujoco.mj_forward(model, data)

print('Camera alignment:', round(aim_camera_once(model, data), 5))
renderer = mujoco.Renderer(model, width=640, height=480)
try:
    sensor = CameraSensor(model, renderer)
    rgb, depth = sensor.capture(data)
    detections = sensor.detect(data, rgb, depth)
    annotated = rgb.copy()
    for label, item in detections.items():
        u, v = item['pixel']
        cv2.drawContours(annotated, [item['contour']], -1, (0, 255, 0), 2)
        cv2.putText(annotated, label, (max(0, int(u)-50), max(20, int(v)-12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imshow('RGB-D detection (press any key)', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    renderer.close()
    raise

blue = detections.get('Yellow Cube')
if blue is None:
    raise SystemExit('Yellow Cube not detected in RGB-D: stopping.')
visible_xyz = np.asarray(blue['position_world_m'], dtype=float)
# Cube is 40 mm tall. The detected top face is about 20 mm above its centre.
# RGB-D provides XY; top/side-face Z can be biased. Known table and cube
# dimensions give a more dependable vertical grasp reference (not cube pose).
center = visible_xyz.copy()
center[2] = 0.021  # 40 mm cube resting on tabletop Z=0, ~1 mm clearance
print('Yellow surface XYZ (RGB-D):', np.round(visible_xyz, 4))
print('Approx. grasp center XYZ:', np.round(center, 4))
if not (.24 <= center[0] <= .46 and -.23 <= center[1] <= .23
        and -.01 <= center[2] <= .07):
    raise SystemExit('RGB-D target outside the pickup area: stopping.')

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

# With the supplied gripper geometry, the fingertips extend 15 mm past the
# TCP in local +Z. Pointing +Z down gives an overhead grasp and vertical lift.
# Keep TCP at cube-centre height +5 mm to reduce tabletop contact risk.
GRASP_Z_OFFSET = .005  # TCP Z=0.026: pads bottom near 0.006, above tabletop
HOVER_ABOVE_CUBE = .20
TRANSFER_HEIGHT = .24  # Clearance above bin walls, not collision avoidance
BIN_RELEASE_HEIGHT = .12  # Cube released from above open bin
GRASP_POSITION_TOL = .0035
GRASP_ORIENTATION_TOL = .07
TABLE_Z = 0.0
MIN_CAMERA_Z = .025   # Safety check on camera position only; NOT full collision detection.



# Live wrist-camera monitor (uses the actual MuJoCo camera attached to wrist_3_link).
# The RGB-D detection above remains a one-time observation; this window shows
# fresh raw RGB frames while the arm moves, not object/body-pose ground truth.
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
                # X closes only the separate camera window, not the robot viewer.
                camera_window_open = False
                return True
        except cv2.error:
            camera_window_open = False
            return True
    else:
        # Do not reopen after the user manually closed the camera window.
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


def down_orientation():
    """Construct target TCP rotation with local +Z pointing vertically down.

    Preserve the horizontal heading of the initial TCP X axis, if possible,
    to avoid an arbitrary 180-degree wrist rotation.
    """
    mujoco.mj_forward(model, data)
    old = data.site_xmat[TCP].reshape(3, 3).copy()
    x = old[:, 0].copy()
    x[2] = 0.
    if np.linalg.norm(x) < 1e-5:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
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


def move_tcp_pose(viewer, target_xyz, target_R, label, max_iters=550,
                  position_tol=.007, velocity_limit=.60):
    """6D damped-Jacobian servo. Abort if a safe, stable pose is not achieved."""
    target_xyz = np.asarray(target_xyz, dtype=float)
    print(f'{label}: target XYZ={np.round(target_xyz, 3)}')
    q_target = np.array(data.qpos[qadr])
    jpos = np.zeros((3, model.nv))
    jrot = np.zeros((3, model.nv))
    best = float('inf')
    stalled = 0
    for i in range(max_iters):
        mujoco.mj_forward(model, data)
        position = data.site_xpos[TCP].copy()
        R = data.site_xmat[TCP].reshape(3, 3)
        pos_err = target_xyz - position
        ori_err = rotation_error(R, target_R)
        pos_norm = float(np.linalg.norm(pos_err))
        ori_norm = float(np.linalg.norm(ori_err))
        # When the tool points down, the sine-based rotation error is small.
        # Also require axis agreement to reject an approximately 180-degree flip.
        z_agreement = float(np.dot(R[:, 2], target_R[:, 2]))
        if pos_norm < position_tol and ori_norm < GRASP_ORIENTATION_TOL and z_agreement > .99:
            print(f'  Reached: XYZ error {pos_norm:.3f} m, tool Z dot {z_agreement:.3f}')
            return True
        combined = pos_norm + .08 * ori_norm
        if combined < best - .00005:
            best, stalled = combined, 0
        else:
            stalled += 1
        # Camera height check prevents blindly moving the camera into the table.
        # This is NOT a collision check on the camera housing, palm or robot.
        if data.cam_xpos[CAM, 2] < MIN_CAMERA_Z:
            print('  STOP: camera centre is too close to/below tabletop.')
            return False
        mujoco.mj_jacSite(model, data, jpos, jrot, TCP)
        J = np.vstack((jpos[:, dadr], jrot[:, dadr]))
        command = np.concatenate((2.0 * pos_err, 1.1 * ori_err))
        damping = .085
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), command)
        dq = np.clip(dq, -velocity_limit, velocity_limit)
        dt_control = max(model.opt.timestep, .04)
        measured = np.asarray(data.qpos[qadr])
        q_target = np.clip(q_target, measured - .12, measured + .12)
        q_target = np.clip(q_target + dq * dt_control, q_min, q_max)
        data.ctrl[:6] = q_target
        if not step_hold(viewer, dt_control):
            return False
        if i % 100 == 0:
            print(f'  iteration {i}: XYZ error={pos_norm:.3f} m, '
                  f'orientation error={ori_norm:.3f}, tool Z dot={z_agreement:.3f}')
        if stalled > 200:
            break
    mujoco.mj_forward(model, data)
    print('  FAILED to converge. TCP=', np.round(data.site_xpos[TCP], 3),
          'camera Z=', round(float(data.cam_xpos[CAM, 2]), 3))
    return False


# MuJoCo ground-truth poses are used ONLY to validate grip after pickup,
# not to calculate the cube pickup target. The drop site is a known bin fixture.
CUBE_BODY = model.body('yellow_cube').id
BIN_DROP = model.site('yellow_bin_drop').id
BIN_BODY = model.body('yellow_bin').id


def carried_cube_state():
    """Report actual cube state for evaluation, not for grasp localisation."""
    mujoco.mj_forward(model, data)
    cube_pos = data.xpos[CUBE_BODY].copy()
    tcp_pos = data.site_xpos[TCP].copy()
    horizontal_error = float(np.linalg.norm(cube_pos[:2] - tcp_pos[:2]))
    return cube_pos, horizontal_error


# Validation ONLY: initial cube pose comes from MuJoCo, never used by IK.
INITIAL_CUBE_Z = float(data.xpos[CUBE_BODY, 2])
LAST_REL_Z = None

def finger_contacts():
    """Count cube-pad contacts reported by MuJoCo contact solver."""
    cube_geom = model.geom('yellow_cube_geom').id
    pad_ids = {model.geom('finger_left_pad').id, model.geom('finger_right_pad').id}
    touching = set()
    for c in data.contact:
        ids = {int(c.geom1), int(c.geom2)}
        if cube_geom in ids:
            touching.update(ids & pad_ids)
    return len(touching)

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
    if contacts == 0:
        print('  STOP: neither finger pad is contacting the cube.')
        return False
    return True


def move_vertical(viewer, xy, z_end, R, title, steps=8, speed=.32,
                  tolerance=.0045):
    mujoco.mj_forward(model, data)
    z_start = float(data.site_xpos[TCP, 2])
    for index, z in enumerate(np.linspace(z_start, z_end, steps + 1)[1:], 1):
        waypoint = np.array([xy[0], xy[1], z], dtype=float)
        if not move_tcp_pose(viewer, waypoint, R, f'{title} {index}/{steps}',
                             max_iters=420, position_tol=tolerance,
                             velocity_limit=speed):
            return False
    return True


try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if not step_hold(viewer, 1.0):
            raise SystemExit('Viewer closed.')

        target_R = down_orientation()
        grasp = center.copy()
        grasp[2] += GRASP_Z_OFFSET
        hover = grasp.copy()
        hover[2] += HOVER_ABOVE_CUBE

        start_xyz = data.site_xpos[TCP].copy()
        if not move_tcp_pose(viewer, start_xyz, target_R,
                             'Point gripper vertically down'):
            raise SystemExit('Stopped: cannot align gripper.')
        if not move_tcp_pose(viewer, hover, target_R,
                             'Hover over yellow cube', velocity_limit=.45):
            raise SystemExit('Stopped: hover unreachable.')

        if not move_vertical(viewer, grasp[:2], grasp[2], target_R,
                             'Descend', speed=.24,
                             tolerance=GRASP_POSITION_TOL):
            raise SystemExit('Stopped: could not descend accurately.')

        data.ctrl[GRIP_CTRL] = FINGER_TRAVEL
        print('Closing fingers and waiting for contact...')
        if not step_hold(viewer, 1.5):
            raise SystemExit('Viewer closed.')

        # Gently raise by 6 cm first. This is the critical slip test.
        mujoco.mj_forward(model, data)
        test_lift_z = float(data.site_xpos[TCP, 2]) + .065
        if not move_vertical(viewer, grasp[:2], test_lift_z, target_R,
                             'Slow initial lift', steps=8, speed=.12,
                             tolerance=GRASP_POSITION_TOL):
            raise SystemExit('Stopped: initial lift failed.')
        if not step_hold(viewer, .35):
            raise SystemExit('Viewer closed.')
        if not check_carry('After initial lift', min_rise=.020):
            raise SystemExit('Stopped: cube slipping. No bin transfer attempted.')

        # Raise farther before translating sideways, to clear the tabletop.
        if not move_vertical(viewer, grasp[:2], TRANSFER_HEIGHT, target_R,
                             'Clear table', steps=7, speed=.28):
            raise SystemExit('Stopped: cannot reach transport height.')
        if not check_carry('Before transfer', min_rise=.065):
            raise SystemExit('Stopped: cube lost before transfer.')

        # Destination is a fixture defined by the XML bin site. The pickup
        # remains fully RGB-D based; no ground-truth cube pose enters control.
        mujoco.mj_forward(model, data)
        bin_xyz = data.site_xpos[BIN_DROP].copy()
        print('Yellow bin drop site (known scene fixture):',
              np.round(bin_xyz, 3))
        above_bin = np.array([bin_xyz[0], bin_xyz[1], TRANSFER_HEIGHT])
        if not move_tcp_pose(viewer, above_bin, target_R,
                             'Transfer above yellow bin', max_iters=700,
                             velocity_limit=.30, position_tol=.008):
            raise SystemExit('Stopped: bin transfer pose unreachable.')
        if not check_carry('Over bin', min_rise=.065):
            raise SystemExit('Stopped: cube lost during transfer.')

        # Stop well above the rim, then release. No attempt to push the cube
        # inside the bin; gravity handles the final drop.
        if not move_vertical(viewer, above_bin[:2], BIN_RELEASE_HEIGHT,
                             target_R, 'Lower over bin', steps=6,
                             speed=.20, tolerance=.005):
            raise SystemExit('Stopped: release height unreachable.')
        if not check_carry('Before release', min_rise=.040):
            raise SystemExit('Stopped: cube lost before release.')

        print('Opening gripper above yellow bin...')
        data.ctrl[GRIP_CTRL] = 0.0
        if not step_hold(viewer, 1.5):
            raise SystemExit('Viewer closed.')
        if not move_vertical(viewer, above_bin[:2], TRANSFER_HEIGHT,
                             target_R, 'Retreat', steps=6, speed=.28):
            print('Warning: released cube but retreat did not finish.')

        if not step_hold(viewer, 1.0):
            raise SystemExit('Viewer closed.')
        mujoco.mj_forward(model, data)
        final_cube = data.xpos[CUBE_BODY].copy()
        final_bin = data.xpos[BIN_BODY].copy()
        # Evaluation only. Bin footprint is 0.11 x 0.11 m.
        inside_xy = bool(np.all(np.abs(final_cube[:2] - final_bin[:2]) < .045))
        inside_z = bool(.005 < final_cube[2] < .075)
        print('Final yellow cube XYZ:', np.round(final_cube, 3))
        print('Final yellow bin XYZ:', np.round(final_bin, 3))
        print('PLACEMENT CHECK:', 'PASS' if inside_xy and inside_z
              else 'NOT CONFIRMED: cube may have missed bin')

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if not update_live_camera():
                break
            time.sleep(model.opt.timestep)
finally:
    cleanup_camera()
