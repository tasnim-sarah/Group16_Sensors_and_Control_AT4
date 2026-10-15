import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

model_path = Path(__file__).parent / "universal_robots_ur3e" / "ur3e.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# --- Look up the 6 arm actuators and joints by name (don't assume ordering) ---
ACT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACT_NAMES]

# End-effector site used for Cartesian targets (Menagerie's UR3e has "attachment_site")
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

# Joint limits for the 6 arm joints (assumes they are the first 6 in qpos, true for this model)
q_min, q_max = model.jnt_range[:6, 0], model.jnt_range[:6, 1]


def solve_ik(target_pos, q_init, iters=300, tol=1e-4, damping=1e-2, step=0.5):
    """Damped least-squares IK for position only (orientation left free).
    Runs on a scratch MjData so the live simulation isn't disturbed."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos          # copy full state (keeps any extra joints)
    ik_data.qpos[:6] = q_init
    jacp = np.zeros((3, model.nv))

    for _ in range(iters):
        mujoco.mj_forward(model, ik_data)
        err = target_pos - ik_data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, None, site_id)
        J = jacp[:, :6]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        ik_data.qpos[:6] = np.clip(ik_data.qpos[:6] + step * dq, q_min, q_max)

    return ik_data.qpos[:6].copy()


# --- Define the targets ---
q_home = data.qpos[:6].copy()

# Option A: joint-space waypoints (radians), if you already know the angles
# q_a = np.array([0.5, -1.2, 1.5, -1.8, -1.57, 0.0])

# Option B: Cartesian waypoints in world coordinates (metres)
mujoco.mj_forward(model, data)
home_xyz = data.site_xpos[site_id].copy()
print("Home end-effector position:", home_xyz)

targets_xyz = [
    home_xyz + np.array([0.10,  0.10, -0.05]),
    home_xyz + np.array([0.10, -0.10, -0.05]),
    home_xyz + np.array([0.00,  0.00,  0.05]),
]

waypoints = [q_home]
q_prev = q_home
for xyz in targets_xyz:
    q_sol = solve_ik(xyz, q_prev)   # seed with previous solution for continuity
    waypoints.append(q_sol)
    q_prev = q_sol
waypoints.append(q_home)            # return home

# --- Motion settings ---
move_time = 3.0     # seconds per segment
pause_time = 0.5    # hold at each waypoint

def smoothstep(u):
    return 3 * u**2 - 2 * u**3      # zero velocity at both ends, avoids jerky starts

with mujoco.viewer.launch_passive(model, data) as viewer:
    seg = 0
    seg_t = 0.0
    dt = model.opt.timestep

    while viewer.is_running():
        step_start = time.time()

        if seg < len(waypoints) - 1:
            q0, q1 = waypoints[seg], waypoints[seg + 1]
            u = min(seg_t / move_time, 1.0)
            q_cmd = q0 + smoothstep(u) * (q1 - q0)
            data.ctrl[act_ids] = q_cmd

            seg_t += dt
            if seg_t >= move_time + pause_time:
                seg += 1
                seg_t = 0.0
                mujoco.mj_forward(model, data)
                print(f"Reached waypoint {seg}, EE at {data.site_xpos[site_id].round(3)}")

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)