"""
SCENE TEST ONLY
"""
import time
from pathlib import Path
 
import mujoco
import mujoco.viewer
import numpy as np
 
HERE = Path(__file__).parent  
 

spec = mujoco.MjSpec.from_file(str(HERE / "universal_robots_ur3e" / "pick_scene.xml"))
 

for key in list(spec.keys):
    spec.delete(key)
 
# Wrist camera
wrist = spec.body("wrist_3_link")
wrist.add_geom(name="camera_shell", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.045, 0.0125, 0.0125],        # 90 x 25 x 25 mm
               pos=[0, 0.08, 0.05], rgba=[0.15, 0.15, 0.15, 1],
               mass=0.072, contype=0, conaffinity=0)
wrist_cam = wrist.add_camera(name="wrist_cam", pos=[0.0325, 0.095, 0.05], fovy=51.28)
wrist_cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
wrist_cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]  

# gripper
gripper = wrist.add_body(name="gripper_base", pos=[0, 0.09215, 0])
gripper.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
gripper.alt.xyaxes = [0, 0, 1, 1, 0, 0] 

gripper.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.045, 0.01],
                 pos=[0, 0, 0.01], mass=0.25, rgba=[0.2, 0.2, 0.2, 1],
                 contype=1, conaffinity=1, group=0)

FINGER_TRAVEL = 0.022 
for side, sign in [("left", 1), ("right", -1)]:
    finger = gripper.add_body(name=f"finger_{side}", pos=[0, sign * 0.032, 0.04])
    finger.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                     axis=[0, -sign, 0], range=[0, FINGER_TRAVEL],
                     damping=5, armature=0.001)
    finger.add_geom(name=f"finger_{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.008, 0.005, 0.02], mass=0.05, rgba=[0.3, 0.3, 0.3, 1],
                    friction=[1.5, 0.01, 0.001], condim=4,
                    contype=1, conaffinity=1, group=0)

# Tool centre point
gripper.add_site(name="tcp", pos=[0, 0, 0.045],size=[0.006, 0, 0], rgba=[1, 0, 1, 1])

# make sure both fingers move together
spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                  name1="finger_right", name2="finger_left")

# Gripper actuator
grip = spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT)
grip.set_to_position(kp=400)
grip.ctrlrange = [0, FINGER_TRAVEL]
grip.forcerange = [-20, 20]

model = spec.compile() 
data = mujoco.MjData(model)
 

# Home pose and random object positions
# Home pose from the UR3e repo 
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
 
rng = np.random.default_rng()
spots = [(0.28, 0.42, 0.05, 0.20),     # x range, y range for each object
         (0.28, 0.42, -0.08, 0.05),
         (0.28, 0.42, -0.20, -0.08)]
rng.shuffle(spots)
for name, (x0, x1, y0, y1) in zip(["red_cube_joint", "yellow_cube_joint", "blue_cube_joint"], spots):
    adr = model.jnt_qposadr[model.joint(name).id]
    data.qpos[adr:adr + 2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
mujoco.mj_forward(model, data)
 
for body in ["red_cube", "yellow_cube", "blue_cube"]:
    print(f"{body:12s} at", np.round(data.body(body).xpos, 3))
print("Tip: Rendering -> Camera -> wrist_cam shows what the robot sees.")
 
# Run
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        data.ctrl[:6] = HOME
        mujoco.mj_step(model, data)
        viewer.sync()
        sleep = model.opt.timestep - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)