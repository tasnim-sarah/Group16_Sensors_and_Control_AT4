"""
Sorting robot: UR3e arm + wrist camera + gripper (MuJoCo).
The camera finds the coloured cubes and trays, the arm picks each cube in the
order you type and drops it into its matching tray.
Run with:  mjpython test_simple.py
"""
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from cameramodulekal import CameraSensor, add_wrist_camera, aim_camera_once

# --------------------------------------------------
# SETTINGS (change these to play around)
# --------------------------------------------------
HOVER_Z = 0.12            # height of the gripper tip when travelling (m)
GRASP_Z = 0.03            # height of the gripper tip when grabbing a cube (m)
DROP_Z = 0.12             # height of the gripper tip when releasing over a tray (m)
FINGER_TRAVEL = 0.034     # how far each finger slides when closing (m)
FINGER_OFFSET = 0.045     # distance of each finger from the centre when open (m)
GRIPPER_OPEN = 0.0        # gripper command: open
GRIPPER_CLOSED = FINGER_TRAVEL   # gripper command: closed
MAX_SPEED = 0.25          # fastest the gripper tip may move (m/s)
PAN_SPEED = 0.4           # speed the base turns towards a tray (rad/s)
CUBE_HALF = 0.02          # half a cube's size (m): the camera sees the front face, so we aim this far deeper
RED_AMPLITUDE = 0.06      # the red cube slides this far left and right (m)
RED_PERIOD = 6.0          # seconds for the red cube to slide there and back
LEAD_TIME = 0.15          # aim this far ahead of the moving red cube (s)
PRINT_EVERY = 6           # print the red cube every 6th camera picture (6 x 1/30 s = 0.2 s)

LOOK_CUBES = np.array([0.35, 0.0, 0.03])   # where the camera looks at the start (x, y, z in m)

CUBES = ["Red Cube", "Yellow Cube", "Blue Cube"]
CUBE_TO_TRAY = {"Red Cube": "Pink Tray", "Yellow Cube": "Green Tray", "Blue Cube": "Purple Tray"}
TRAY_BODY = {"Pink Tray": "red_bin", "Green Tray": "yellow_bin", "Purple Tray": "blue_bin"}
TRAY_XY = {"Pink Tray": (0.32, 0.28), "Green Tray": (0.00, 0.40), "Purple Tray": (0.16, 0.36)}
NAMES = {"red": "Red Cube", "yellow": "Yellow Cube", "blue": "Blue Cube"}

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
    motor.set_to_position(kp=1000)
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
rng = np.random.default_rng()                             # new random numbers every run
strips = [(0.10, 0.15), (-0.025, 0.025), (-0.15, -0.10)]
rng.shuffle(strips)
for joint, (ymin, ymax) in zip(["red_cube_joint", "yellow_cube_joint", "blue_cube_joint"], strips):
    adr = int(model.jnt_qposadr[model.joint(joint).id])   # where this joint starts in data.qpos
    data.qpos[adr] = rng.uniform(0.28, 0.35)              # x (forwards)
    data.qpos[adr + 1] = rng.uniform(ymin, ymax)          # y (sideways)

mujoco.mj_forward(model, data)                            # update everything after our changes
aim_camera_once(model, data, LOOK_CUBES)                  # tilt the camera towards the cubes

# remember a few things about the arm
BASE_XY = data.xanchor[0][:2].copy()                      # x, y of the base joint
site_id = model.site("tcp").id
FINGER_ADR = int(model.jnt_qposadr[model.joint("finger_left").id])
R_HOME = data.site_xmat[site_id].reshape(3, 3).copy()     # the direction the gripper points at HOME
R_des = R_HOME.copy()                                     # the direction we want to keep
P_HOME = data.site_xpos[site_id].copy()                   # tcp position at HOME

# the red cube slides, so remember its joint and its starting y
red_joint = model.joint("red_cube_joint")
red_qadr = int(model.jnt_qposadr[red_joint.id])
red_dadr = int(model.jnt_dofadr[red_joint.id])
red_y0 = float(data.qpos[red_qadr + 1])
red_moving = True

q_cmd = HOME.copy()              # joint angles we ask the arm to go to
gripper_cmd = GRIPPER_OPEN       # gripper command we ask for
jacp = np.zeros((3, model.nv))   # space for the Jacobian (position part)
jacr = np.zeros((3, model.nv))   # space for the Jacobian (rotation part)
dt = model.opt.timestep          # length of one physics step (s)
DETECT_EVERY = round(1 / 30 / dt)   # take a camera picture every this many steps (30 per second)

# --------------------------------------------------
# 2. CAMERA TRACKING
# --------------------------------------------------
renderer = mujoco.Renderer(model, width=640, height=480)
sensor = CameraSensor(model, renderer)   # from cameramodulekal.py: finds colours + Kalman filter

tracks = {}      # name -> last known position of that object (x, y, z)
velocity = {}    # name -> speed of that object from the Kalman filter
target = None    # the object we are working on now: a cube name or a tray name


def look_at():
    """Where the camera should point right now."""
    if target in tracks:
        return tracks[target]       # where we last saw the target
    if target in BIN_AREA:
        return BIN_AREA[target]     # a tray we have not seen yet: use its rough spot
    return LOOK_CUBES


def update_tracks():
    """Take one camera picture and update where everything is."""
    aim_camera_once(model, data, look_at())           # point the camera
    rgb, depth = sensor.capture(data)                 # take a colour and a depth picture
    found = sensor.detect(data, rgb, depth)           # find the coloured objects and smooth them (Kalman)
    for name in found:
        pos = found[name]["position_world_m"].copy()
        if name in CUBES:                             # the camera sees the front face of a cube,
            ray = pos - data.cam_xpos[sensor.cam_id]  # so move the point half a cube further away
            pos = pos + CUBE_HALF * ray / np.linalg.norm(ray)
        tracks[name] = pos
        velocity[name] = sensor.position_filters[name].kf.statePost[3:6, 0].astype(float)
    for name in tracks:
        if name not in found:                         # not visible now: assume it is not moving
            velocity[name] = np.zeros(3)
    return found


def cube_xy(name):
    """x, y where the gripper should go for this cube."""
    pos = tracks[name]
    if name == "Red Cube" and red_moving:
        pos = pos + velocity[name] * LEAD_TIME        # aim a little ahead of the moving red cube
    return pos[:2]


# --------------------------------------------------
# 3. MOVING THE ARM
# --------------------------------------------------
def servo(point):
    """Nudge the arm one small step so the gripper tip moves towards `point`."""
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])         # Jacobian: how joint speeds change the tip's speed

    vel = 5.0 * (point - data.site_xpos[site_id])     # move faster when further away
    if np.linalg.norm(vel) > MAX_SPEED:               # ...but never faster than MAX_SPEED
        vel *= MAX_SPEED / np.linalg.norm(vel)

    R = data.site_xmat[site_id].reshape(3, 3)         # which way the gripper points now
    rot_err = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))   # how far it is from the wanted direction

    twist = np.r_[vel, 5.0 * rot_err]                 # wanted motion: move + keep pointing down
    J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))   # "inverse" of the Jacobian
    dq = J_inv @ twist                                # joint speeds that give that motion
    posture = np.r_[0.0, HOME[1:6] - data.qpos[1:6]]  # gently pull the elbow and wrists back to HOME
    dq += (np.eye(6) - J_inv @ J) @ posture           # ...without disturbing the tip

    # new joint targets, never too far ahead of where the arm really is
    q_cmd[:] = np.clip(q_cmd + dq * dt, data.qpos[:6] - 0.3, data.qpos[:6] + 0.3)


def wrap(angle):
    """Turn any angle into the range -pi to +pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def azimuth(xy):
    """The angle of a point as seen from the base."""
    d = np.asarray(xy) - BASE_XY
    return np.arctan2(d[1], d[0])


step = 0


def step_sim():
    """Run ONE physics step: slide the red cube, take a camera picture, move the motors."""
    global step
    if not viewer.is_running():                       # window closed: stop the program
        raise SystemExit

    if red_moving:                                    # red cube slides left and right
        w = 2 * np.pi / RED_PERIOD
        data.qpos[red_qadr + 1] = red_y0 + RED_AMPLITUDE * np.sin(w * data.time)
        data.qvel[red_dadr + 1] = RED_AMPLITUDE * w * np.cos(w * data.time)

    if step % DETECT_EVERY == 0:                      # 30 times a second
        found = update_tracks()
        if red_moving and (step // DETECT_EVERY) % PRINT_EVERY == 0:   # print the red cube live
            if "Red Cube" in found:
                red = found["Red Cube"]
                print(f"t={data.time:5.2f}s  camera {np.round(red['position_world_raw_m'], 3)}  "
                      f"kalman {np.round(red['position_world_m'], 3)}  "
                      f"velocity {np.round(velocity['Red Cube'], 3)}")
            else:
                print(f"t={data.time:5.2f}s  Red Cube not in view")
        viewer.sync()                                 # redraw the window
        time.sleep(1 / 30)                            # keep it near real time

    data.ctrl[:6] = q_cmd                             # send the joint targets to the arm motors
    data.ctrl[6] = gripper_cmd                        # send the gripper command
    mujoco.mj_step(model, data)                       # move the physics forward one step
    step += 1


def goto(get_target, tol=0.01):
    """Move the gripper tip to the point returned by get_target() (we ask again each step, so it can follow a moving cube)."""
    start = data.time
    while data.time - start < 12:                     # give up after 12 seconds
        point = get_target()
        if np.linalg.norm(data.site_xpos[site_id] - point) < tol:
            break
        servo(point)
        step_sim()


def wait(seconds):
    """Hold the gripper still for a while."""
    start = data.time
    while data.time - start < seconds:
        servo(data.site_xpos[site_id].copy())
        step_sim()


def rotate_base(pan_target):
    """Turn only the base joint to a new angle, then remember the new gripper direction."""
    global R_des
    while abs(pan_target - q_cmd[0]) > 0.01:
        q_cmd[0] += np.clip(pan_target - q_cmd[0], -PAN_SPEED * dt, PAN_SPEED * dt)
        step_sim()
    start = data.time
    while data.time - start < 0.5:                    # let the arm settle
        step_sim()
    R_des = data.site_xmat[site_id].reshape(3, 3).copy()


# --------------------------------------------------
# 4. PICK AND PLACE
# --------------------------------------------------
def pick_and_place(cube, tray):
    global gripper_cmd, red_moving, target, R_des
    print(f"\n>>> Picking {cube} -> {tray}")
    target = cube                                     # the camera now looks at this cube
    while cube not in tracks:                         # wait until the camera has seen it
        step_sim()

    gripper_cmd = GRIPPER_OPEN
    goto(lambda: np.r_[cube_xy(cube), HOVER_Z], tol=0.012)       # 1. fly above the cube (follows it)

    if cube == "Red Cube":                                       # the red cube stops sliding
        red_moving = False
        data.qvel[red_dadr:red_dadr + 6] = 0
    wait(0.5)                                                    # let the camera settle on a still cube
    spot = cube_xy(cube).copy()                                  # remember where the cube is

    goto(lambda: np.r_[spot, GRASP_Z], tol=0.008)                # 2. go straight down
    gripper_cmd = GRIPPER_CLOSED                                 # 3. close the gripper
    wait(0.8)
    goto(lambda: np.r_[data.site_xpos[site_id][:2], HOVER_Z], tol=0.015)   # 4. lift up
    if data.qpos[FINGER_ADR] < FINGER_TRAVEL - 0.005:            # fingers stopped early = holding a cube
        print("  holding the cube")
    else:
        print("  MISSED the cube")

    target = tray                                                # the camera now looks at the tray
    tracks.pop(tray, None)                                       # forget old tray position
    sensor.position_filters[tray].reset()                        # start the Kalman filter fresh
    turn = wrap(azimuth(BIN_AREA[tray][:2]) - azimuth(data.site_xpos[site_id][:2]))
    rotate_base(q_cmd[0] + turn)                                 # 5. turn towards the tray

    start = data.time
    while tray not in tracks and data.time - start < 3:          # wait up to 3 s for the camera to find it
        step_sim()
    if tray not in tracks:
        print(f"  {tray} not seen - using its rough position")
        tracks[tray] = BIN_AREA[tray].copy()

    goto(lambda: np.r_[tracks[tray][:2], DROP_Z], tol=0.012)     # 6. move over the tray
    gripper_cmd = GRIPPER_OPEN                                   # ...and let go
    wait(0.7)

    goto(lambda: np.r_[data.site_xpos[site_id][:2], DROP_Z], tol=0.015)   # 7. up, turn back, go home
    rotate_base(HOME[0])
    R_des = R_HOME.copy()
    target = None
    for name in CUBES:                                           # cubes were out of view: start their filters fresh
        sensor.position_filters[name].reset()
    goto(lambda: P_HOME, tol=0.02)
    print(f"<<< {cube} done")


# --------------------------------------------------
# 5. RUN
# --------------------------------------------------
words = input("Pick order (e.g. 'red blue yellow'): ").lower().split()
order = [NAMES[word] for word in words]
print("Order:", order)

viewer = mujoco.viewer.launch_passive(model, data)    # open the window
for cube in order:
    pick_and_place(cube, CUBE_TO_TRAY[cube])
print("\nAll cubes sorted.")

renderer.close()
viewer.close()