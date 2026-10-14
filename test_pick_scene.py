"""
SCENE TEST ONLY
"""
import time
 
import mujoco
import mujoco.viewer
import numpy as np
 
model = mujoco.MjModel.from_xml_path("pick_scene.xml")
data = mujoco.MjData(model)
 
# Home pose from the falque UR3e repo (pan, lift, elbow, wrist1, wrist2, wrist3)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
 
# Random object positions on the table, kept apart so they don't overlap
rng = np.random.default_rng()
spots = [(0.28, 0.42, 0.05, 0.20),     # x range, y range for each object
         (0.28, 0.42, -0.08, 0.05),
         (0.28, 0.42, -0.20, -0.08)]
rng.shuffle(spots)
for name, (x0, x1, y0, y1) in zip(["cube_joint", "sphere_joint", "pyramid_joint"], spots):
    adr = model.jnt_qposadr[model.joint(name).id]
    data.qpos[adr:adr + 2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
mujoco.mj_forward(model, data)
 
for body in ["cube", "sphere", "pyramid"]:
    print(f"{body:8s} at", np.round(data.body(body).xpos, 3))
 
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        data.ctrl[:6] = HOME
        mujoco.mj_step(model, data)
        viewer.sync()
        sleep = model.opt.timestep - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)
 