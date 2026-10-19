"""Blue-cube RGB-D pickup experiment, separate from the team's main scene.

Requires: pick_scene.xml with all 3 cubes and bins, camera_module.py, MuJoCo,
OpenCV and NumPy. Detection uses RGB-D rather than the cube's MuJoCo body pose.
The physical grasp is experimental and may need camera/IK/gripper tuning.
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
               size=[.008, .005, .02], mass=.05, rgba=[.3, .3, .3, 1],
               friction=[1.5, .01, .001], condim=4,
               contype=1, conaffinity=1, group=0)
g.add_site(name='tcp', pos=[0, 0, .045], size=[.006, 0, 0], rgba=[1, 0, 1, 1])
spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT,
                  objtype=mujoco.mjtObj.mjOBJ_JOINT,
                  name1='finger_right', name2='finger_left')
a = spec.add_actuator(name='gripper', target='finger_left',
                      trntype=mujoco.mjtTrn.mjTRN_JOINT)
a.set_to_position(kp=400)
a.ctrlrange = [0, FINGER_TRAVEL]
a.forcerange = [-20, 20]

model = spec.compile()
data = mujoco.MjData(model)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
# Original non-overlapping placement bands (one cube per band).
rng = np.random.default_rng()
bands = [(0.28, .42, .05, .20), (.28, .42, -.08, .05),
         (.28, .42, -.20, -.08)]
rng.shuffle(bands)
for name, (x0, x1, y0, y1) in zip(
        ['red_cube_joint', 'yellow_cube_joint', 'blue_cube_joint'], bands):
    adr = int(model.jnt_qposadr[model.joint(name).id])
    data.qpos[adr:adr+2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
# First open the gripper (0=wide, 0.022=closed on both fingers).
GRIP_CTRL = model.actuator('gripper').id
data.ctrl[GRIP_CTRL] = 0.
mujoco.mj_forward(model, data)

# Set the camera's mounting direction ONCE at the observation pose.
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
finally:
    renderer.close()

blue = detections.get('Blue Cube')
if blue is None:
    raise SystemExit('Blue Cube not found in RGB-D frame. No pickup attempted. '
                     'Check the wrist camera view and HSV thresholds.')
# The depth point is on the visible cube SURFACE; for a 40 mm cube,
# approximate cube CENTER by lowering a top-face measurement by 20 mm.
# This assumes the wrist camera sees the top face (verify in the preview).
visible_xyz = np.array(blue['position_world_m'], dtype=float)
center = visible_xyz + np.array([0., 0., -.020])
print('Blue surface XYZ (RGB-D):', np.round(visible_xyz, 4))
print('Approx. grasp center XYZ:', np.round(center, 4))
if not (.24 <= center[0] <= .46 and -.23 <= center[1] <= .23
        and -.01 <= center[2] <= .07):
    raise SystemExit('Detected target is outside safe test pickup bounds; stopping.')

TCP = model.site('tcp').id
# Joint-space Jacobian controller, no access to true cube location.
joint_ids = [model.joint(n).id for n in
             ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
qadr = [int(model.jnt_qposadr[j]) for j in joint_ids]
dadr = [int(model.jnt_dofadr[j]) for j in joint_ids]
q_min = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -2*np.pi
                  for j in joint_ids])
q_max = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else 2*np.pi
                  for j in joint_ids])
# Some UR3e XML joint names vary. Fail with useful explanation if so.

def step_hold(viewer, duration):
    for _ in range(max(1, int(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)
        viewer.sync()
        if not viewer.is_running():
            return False
    return True


def move_tcp(viewer, goal, label, max_iters=600):
    """Position-only resolved-rate control, with diagnostics for failed moves.

    Control targets are integrated explicitly instead of resetting to the
    measured joint angle at every iteration (which can stall under lag).
    This prototype is not collision-aware and does not enforce tool orientation.
    """
    goal = np.asarray(goal, dtype=float)
    print(f'{label}: target {np.round(goal, 3)}')
    mujoco.mj_forward(model, data)
    initial_error = np.linalg.norm(goal - data.site_xpos[TCP])
    print('  Initial TCP:', np.round(data.site_xpos[TCP], 3),
          '| distance:', f'{initial_error:.3f} m')
    jpos = np.zeros((3, model.nv))
    jrot = np.zeros((3, model.nv))
    q_target = np.array([data.qpos[k] for k in qadr])
    best_error = float('inf')
    stalled = 0
    for i in range(max_iters):
        mujoco.mj_forward(model, data)
        err = goal - data.site_xpos[TCP]
        distance = float(np.linalg.norm(err))
        if distance < .012:
            print(f'  Reached with {distance:.3f} m error, iteration {i}')
            return True
        if distance < best_error - 0.0001:
            best_error = distance
            stalled = 0
        else:
            stalled += 1
        mujoco.mj_jacSite(model, data, jpos, jrot, TCP)
        J = jpos[:, dadr]
        # Velocity command in radians per second; normalize to limit each joint.
        lam = .055
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), 2.0 * err)
        dq = np.clip(dq, -.75, .75)
        # Advance the desired joint angles with the simulation's control interval.
        dt_control = max(model.opt.timestep, .04)
        measured = np.array([data.qpos[k] for k in qadr])
        # Avoid letting the requested joint pose outrun physical motion.
        q_target = np.clip(q_target, measured - .12, measured + .12)
        q_target = np.clip(q_target + dt_control * dq, q_min, q_max)
        data.ctrl[:6] = q_target
        if not step_hold(viewer, dt_control):
            return False
        if i % 100 == 0:
            print(f'  iteration {i:3d}: error={distance:.3f} m, '
                  f'TCP={np.round(data.site_xpos[TCP], 3)}')
        if stalled >= 180:
            print('  Progress has stalled; likely joint limits, singularity, '
                  'orientation-related reachability, or control mismatch.')
            break
    mujoco.mj_forward(model, data)
    final_error = float(np.linalg.norm(goal - data.site_xpos[TCP]))
    print(f'  FAILED: initial error {initial_error:.3f} m; '
          f'final error {final_error:.3f} m; best error {best_error:.3f} m')
    print('  Final TCP:', np.round(data.site_xpos[TCP], 3))
    print('  Joint angles (deg):', np.round(np.degrees(data.qpos[qadr]), 1))
    print('  Check camera image and robot pose before attempting grasp.')
    return False

# Only moves when the viewer is active. This is a position-only IK prototype:
# it does not constrain the gripper's orientation or plan around obstacles.
with mujoco.viewer.launch_passive(model, data) as viewer:
    if not step_hold(viewer, 1.0):
        raise SystemExit()
    approach = center + [0, 0, .14]
    # Position-only IK/gripper orientation still needs validating before contact.
    if not move_tcp(viewer, approach, 'Hover above blue cube'):
        raise SystemExit('Stopped: unable to reach hover.')
    if not move_tcp(viewer, center, 'Lower for grasp'):
        raise SystemExit('Stopped: unable to reach grasp.')
    data.ctrl[GRIP_CTRL] = FINGER_TRAVEL
    print('Closing gripper...')
    if not step_hold(viewer, 1.2):
        raise SystemExit()
    if not move_tcp(viewer, approach, 'Lift'):
        raise SystemExit('Lift target not reached.')
    print('Lift finished. Inspect whether the blue cube actually moved with the fingers.')
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
