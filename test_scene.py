"""
SCENE TEST ONLY - not part of the pipeline.
Holds the arm at 'home' using GROUND-TRUTH joint states so the scene can be
inspected. The real controller must use the ESTIMATED state.
Run from the repo folder:  python test_scene.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
q_home = data.qpos[:6].copy()

rng = np.random.default_rng()
cube_adr = model.jnt_qposadr[model.joint("cube_joint").id]
data.qpos[cube_adr:cube_adr + 2] = [rng.uniform(0.40, 0.60), rng.uniform(-0.10, 0.15)]
mujoco.mj_forward(model, data)
print("Cube placed at x, y, z =", np.round(data.qpos[cube_adr:cube_adr + 3], 3))

KP = np.array([200, 200, 100, 30, 30, 30.0])
KD = np.array([20, 20, 10, 3, 3, 3.0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        q, qd = data.qpos[:6], data.qvel[:6]
        data.ctrl[:6] = data.qfrc_bias[:6] + KP * (q_home - q) - KD * qd
        mujoco.mj_step(model, data)
        viewer.sync()
        sleep = model.opt.timestep - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)