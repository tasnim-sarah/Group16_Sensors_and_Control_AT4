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
 
model = spec.compile()
data = mujoco.MjData(model)
 

# Home pose and random object positions
# Home pose from the UR3e repo 
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
 
rng = np.random.default_rng()
spots = [(0.28, 0.42, 0.05, 0.20),    
         (0.28, 0.42, -0.08, 0.05),
         (0.28, 0.42, -0.20, -0.08)]
rng.shuffle(spots)
for name, (x0, x1, y0, y1) in zip(["cube_joint", "sphere_joint", "pyramid_joint"], spots):
    adr = model.jnt_qposadr[model.joint(name).id]
    data.qpos[adr:adr + 2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
mujoco.mj_forward(model, data)
 
for body in ["cube", "sphere", "pyramid"]:
    print(f"{body:8s} at", np.round(data.body(body).xpos, 3))
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