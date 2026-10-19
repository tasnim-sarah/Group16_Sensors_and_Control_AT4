import time
import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

model_path = Path(__file__).parent / "universal_robots_ur3e" / "ur3e.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

def act(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise ValueError(f"Actuator '{name}' not found")
    return i

def safe_target(actuator_id, value, margin=0.05):
    """Clamp a target inside the joint's range, with a small safety margin."""
    j = model.actuator_trnid[actuator_id, 0]
    lo, hi = model.jnt_range[j]
    return float(np.clip(value, lo + margin, hi - margin))

tilt_id = act("wrist_1")   # up/down  (swap names if your model moves the other way)
pan_id  = act("wrist_2")   # left/right

home_ctrl = data.ctrl.copy()

# --- Sweep settings ---
amp = 0.5            # radians either side of home (~29 deg)
sweep_period = 6.0   # seconds for one full out-and-back sweep
n_sweeps = 2         # out-and-back sweeps per phase
hold_time = 2.0      # seconds paused at home between phases

# --- The sequence: (name, joint to move or None, duration) ---
sweep_time = sweep_period * n_sweeps
phases = [
    ("up_down",    tilt_id, sweep_time),
    ("home_hold1", None,    hold_time),
    ("left_right", pan_id,  sweep_time),
    ("home_hold2", None,    hold_time),
]
cycle_time = sum(p[2] for p in phases)

def commanded_targets(t):
    """Return the full set of control targets for simulation time t."""
    ctrl = home_ctrl.copy()                 # everything defaults to home
    t_in_cycle = t % cycle_time             # wraps around so the sequence repeats

    start = 0.0
    for name, joint_id, duration in phases:
        if t_in_cycle < start + duration:
            if joint_id is not None:
                local_t = t_in_cycle - start
                offset = amp * math.sin(2 * math.pi * local_t / sweep_period)
                ctrl[joint_id] = safe_target(joint_id, home_ctrl[joint_id] + offset)
            return ctrl
        start += duration
    return ctrl

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[:] = commanded_targets(data.time)

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)