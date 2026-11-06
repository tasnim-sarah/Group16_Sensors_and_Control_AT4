"""Scene construction only: simulator truth is used here to initialise trials."""
from pathlib import Path
import mujoco
import numpy as np
from cameramodulekal import add_wrist_camera, aim_camera_once

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
FINGER_TRAVEL = 0.034
FINGER_OFFSET = 0.045
TRAYS = {"Red Cube": "Red Tray", "Yellow Cube": "Yellow Tray", "Blue Cube": "Blue Tray"}
TRAY_BODY = {"Red Tray": "red_bin", "Yellow Tray": "yellow_bin", "Blue Tray": "blue_bin"}
TRAY_XY = {"Red Tray": (0.32, 0.28), "Yellow Tray": (0.00, 0.40), "Blue Tray": (0.16, 0.36)}

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
                        size=[0.008, 0.005, 0.03], mass=0.05, rgba=[0.3, 0.3, 0.3, 1],
                        friction=[1.5, 0.01, 0.001], condim=4, contype=1, conaffinity=1,
                        solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    # the tool centre point (tcp): the spot between the fingertips that we move around
    gripper.add_site(name="tcp", pos=[0, 0, 0.045], size=[0.006, 0, 0], rgba=[1, 0, 1, 1])

    # the right finger always copies the left finger, so one motor closes both
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      name1="finger_right", name2="finger_left")
    motor = spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    motor.set_to_position(kp=2000)
    motor.ctrlrange = [0, FINGER_TRAVEL]
    motor.forcerange = [-30, 30]

def add_motion_platform(spec):
    """Physical XY stage; moving contacts carry the cube without pose overrides."""
    platform = spec.worldbody.add_body(name='red_motion_platform',pos=[0,0,.0015])
    for axis, direction in [('x',[1,0,0]),('y',[0,1,0])]:
        name='platform_'+axis
        platform.add_joint(name=name,type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=direction,range=[-.12,.12],damping=10)
        motor=spec.add_actuator(name=name,target=name,trntype=mujoco.mjtTrn.mjTRN_JOINT)
        motor.set_to_position(kp=500)
        motor.ctrlrange=[-.12,.12]
        motor.forcerange=[-10,10]
    platform.add_geom(name='motion_platform_surface',type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[.032,.032,.001],mass=.2,rgba=[.35,.35,.35,1],friction=[1.5,.01,.001],
        condim=4,solref=[.004,1],solimp=[.95,.99,.001,.5,2])


def platform_target(time_s, amplitude, period):
    """Side-to-side Y motion, +/- amplitude, with smooth natural reversals."""
    phase=2*np.pi*float(time_s)/float(period)
    return np.array([0.,float(amplitude)*np.sin(phase)])


def build_scene(seed=7, moving_red=False):
    path = Path(__file__).resolve().parent / "universal_robots_ur3e" / "pick_scene.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    for key in list(spec.keys):
        spec.delete(key)
    add_wrist_camera(spec)
    add_gripper(spec)
    if moving_red:
        add_motion_platform(spec)
    model = spec.compile()
    # Frictional grasp contact, without attachment/weld constraints.
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10
    model.opt.iterations = 100
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    for tray, xy in TRAY_XY.items():
        model.body_pos[model.body(TRAY_BODY[tray]).id, :2] = xy
        colour = tray.split()[0].lower()
        palette = model.geom_rgba[model.geom(colour+'_cube_geom').id].copy()
        for side in ('wall_px','wall_nx','wall_py','wall_ny'):
            model.geom_rgba[model.geom(colour+'_bin_'+side).id] = palette
        # A neutral floor separates the coloured cube from the coloured rim in RGB.
        model.geom_rgba[model.geom(colour+'_bin_floor').id] = [.45,.45,.45,1]
    rng = np.random.default_rng(seed)
    strips = [(0.10, 0.15), (-0.025, 0.025), (-0.15, -0.10)]
    rng.shuffle(strips)
    for colour, strip in zip(("red", "yellow", "blue"), strips):
        adr = int(model.jnt_qposadr[model.joint(colour + "_cube_joint").id])
        data.qpos[adr:adr+2] = [rng.uniform(0.28, 0.35), rng.uniform(*strip)]
    if moving_red:
        red_adr=int(model.jnt_qposadr[model.joint('red_cube_joint').id])
        model.body_pos[model.body('red_motion_platform').id,:2]=data.qpos[red_adr:red_adr+2]
        data.qpos[red_adr+2]=.0225  # 2.5 mm platform top + 20 mm cube half-height.
    mujoco.mj_forward(model, data)
    # Fixed mounting calibrated once at a known setup pose. Never re-aim during a trial.
    tcp = data.site_xpos[model.site('tcp').id].copy()
    wrist = model.body('wrist_3_link').id
    # Look between the fingers, perpendicular to their closing axis. A view
    # along the closing axis hides a moving cube behind the nearer finger.
    desired_camera = tcp + np.array([0.03, 0.17, 0.12])
    camera_id = model.camera('wrist_cam').id
    model.cam_pos[camera_id] = data.xmat[wrist].reshape(3,3).T @ (desired_camera-data.xpos[wrist])
    model.cam_fovy[camera_id] = 75.0
    aim_camera_once(model, data, tcp + np.array([0., 0., -0.15]))
    return model, data
