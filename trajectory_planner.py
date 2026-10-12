import mujoco
import numpy as np

def get_jointplacement(model, joint_names):
    qpos_addresses = []
    dof_addresses = []    

    for name in joint_names:
	    id_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
	    if id_joint == -1:
            raise ValueError(f"Joint not found:{name}" )
	    qpos_addresses.append(model.jnt_qposadr[id_joint])
	    dof_addresses.append(model.jnt_dofadr[id_joint])
    qpos_addresses = np.array(qpos_addresses)
    dof_addresses = np.array(dof_addresses)
    current_joint_angles = data.qpos[qpos_addresses].copy()

    # Read the current joint positions using the addresses
    current_joint_angles = data.qpos[qpos_addresses].copy()
    return np.array(qpos_addresses), np.array(dof_addresses) current_joint_angles
    

def ik_solver(model, data, target_pos, joint_names, site_name = "attachment_site", tolerance=1e-3, max_steps = 100, damping= 1e-3):
    qpos_addresses, dof_addresses = get_jointplacement(model, joint_names)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site not found: {site_name}")

	target_pos = np.asarray(target_pos, dtype=float) # convert the target position into a NumPy array
    if target_pos.shape != (3,):
         raise ValueError("target_pos must have [x, y, z]") # checks whether the target has x, y, and z coordinates

	jacobian_position = np.zeros((3, model.nv)) # create an array to store the position jacobian
    jacobian_rotation = np.zeros((3, model.nv)) # create an array to store the rotational jacobian

	for iteration in range(max_steps): 
         mujoco.mj_forward(model, data)
         current_pos = data.site_xpos[site_id].copy() # read the current end effector position in world coordinates
         error = target_pos - current_pos  # measures the position error
         if np.linalg.norm(error) < tolerance: 
             return data.qpos[qpos_addresses].copy()
         mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)  #calculate the jacobian of the end effector site
         J = jacobian_position[:, dof_addresses] #keeps only the columns that are needed by selected arm joints
         Ds = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), error)  # calculates the damped least square joint changes
         data.qpos[qpos_addresses] += Ds
         
  mujoco.mj_forward(model, data) # update the end effector position after the final run
  final_error = np.linalg.norm(target_pos - data.site_xpos[site_id]) 
  if final_error >= tolerance: 
      raise RuntimeError(f"IK did not converge. Final position error: {final_error:.6f} m")
  return data.qpos[qpos_addresses].copy()

  def joint_trajectory(q_start, q_goal, duration,timestep):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if q_start.shape != q_goal.shape:
        raise ValueError("Start and goal must have the same shape.")
    if duration <= 0 or  timestep <= 0:
        raise ValueError("Duration and timestep must be positive.")
    steps = int(np.ceil)


def make_joint_trajectory(q_start, q_goal, duration, timestep):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if q_start.shape != q_goal.shape:
        raise ValueError("Start and goal must have the same shape.")

    if duration <= 0 or timestep <= 0:
        raise ValueError("Duration and timestep must be positive.")
    steps = int(np.ceil(duration / timestep)) + 1
    times = np.linspace(0,duration, steps)

    trajectory = []
    for t in times: 
        s = t/ duration 
        blend = 3 * s**2 - 2* s**3
        q_wanted = q_start + blend * (q_goal - q_start)
        trajectory.append(q_wanted)
    return times, np.array(trajectory)


 

