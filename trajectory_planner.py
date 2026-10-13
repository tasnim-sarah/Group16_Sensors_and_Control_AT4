import mujoco
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
# finds the robot's joint location inside MUjoco's arrays.
# it also gets the joints' current angles 
def get_joint_addresses(model, joint_names, data): # finds the current joint angles of the pose 
    qpos_addresses = []
    dof_addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) # finds the joint's id
        if joint_id  == -1:
            raise ValueError(f"Joint not found:{name}") # checks if joint exists 
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    qpos_addresses = np.array(qpos_addresses, dtype = int)
    dof_addresses = np.array(dof_addresses, dtype = int)
    current_joint_angles = data.qpos[qpos_addresses].copy()
    return qpos_addresses, dof_addresses, current_joint_angles

def actuator_ids(model,joint_names): # actuator uses joint angles to move corresponding links in mujoco
    actuator_ids = []
    for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id  == -1:
                raise ValueError(f"Joint not found:{name}")
            matches = []
            for actuator_id in range(model.nu):
                is_joint_actuator = (model.actuator_trntype[actuator_id] == mujoco.mjttrn.mjTRN_JOINT)
                controls_this_joint = (model.actuator_trnid[actuator_id,0] == joint_id)
                if is_joint_actuator and controls_this_joint:
                    matches.append(actuator_id)
            if len(matches) != 1:
                raise ValueError(f"Expected one exact joint actuator for {name},but found {len(matches)}")
            actuator_ids.append(matches[0])
    return np.array(actuator_ids, dtype=int)

# finds the joint angles that place the end effector at a specified position.
#how the joint angles change over time from current configuration to the target configuration
def solve_ik(robot, target_position, q_current, damping = 0.01, max_steps = 100, tolerance = 1e-6):
    target_position = np.asarray(target_position, dtype=float)
    q_current = np.asarray(q_current, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")
    if q_current.shape != (robot.n,):
        raise ValueError("each robot joint must have a corresponding angle in q_current")
    if not np.all(np.isfinite(target_position)):
        raise ValueError("target_position needs finite values.")
    if not np.all(np.isfinite(q_current)):
        raise ValueError("q_current needs finite values.")
    if damping <= 0:
        raise ValueError("damping needs to be positive")
    if max_steps < 1:
        raise ValueError("max_steps need to be atleast 1")
    if tolerance <= 0: 
        raise ValueError("tolerance should be positive")
    target_pose = SE3(target_position[0], target_position[1], target_position[2])
    result = robot.ikine_LM(target_pose, q0=q_current, mask=[1,1,1,0,0,0],method="chan", k = damping, slimit = max_steps, tol = tolerance)
    if not result.success:
        raise RuntimeError("Ik did not converge. check if target is reachable.")
    return result.q




    





def make_trajectory(q_start,q_goal, duration, timestep):
    q_start = np.asarray(q_start, dtype = float)
    q_goal = np.asarray(q_goal, dtype = float)
    if q_start.shape != q_goal.shape:
        raise ValueError("start and goal need the same number of joints")
    if q_start.ndim != 1:
        raise ValueError("joint confirgurations should be 1d arrays")
    if duration <= 0 or timestep <= 0:
        raise ValueError("duration and timestep should be positive")
    steps = int(np.ceil(duration/timestep))+1
    times = np.linspace(0,duration, steps)
    trajectory = rtb.jtraj(q_start, q_goal, times).q
    return times, trajectory

def execute_trajectory(model, data, actuator_ids, times, trajectory):
    physical_timestep = model.opt.timestep
    total_steps = int(np.ceil(times[-1]/physical_timestep))
    for step in range(total_steps + 1):
        simulation_time = min(step * physical_timestep, times[-1])
        wanted_angles = np.array([np.interp(simulation_time, times, trajectory[:,joint]) 
        for joint in range(trajectory.shape[1])])
        data.ctrl[actuator_ids] = wanted_angles
        if step < total_steps: 
            mujoco.mj_step(model,data)

# uses the camera's instrinsic calibration paramters and depth to calculate a 3d point. world frame

def pixel_to_camerapoint(u,v,depth,fx,fy,cx,cy):
    xcamera = (u- cx) * depth/ fx
    ycamera = (v- cy) * depth / fy 
    zcamera = depth 
    return np.array([xcamera,ycamera,zcamera])
#transforms that point into the robot's base frame. 
def cameratobase(point_camera, Tbasecamera):
    homogenous_point  = np.append(point_camera, 1)
    point_base_homogenous = T_base_camera @ homogenous_point
    return point_base_homogenous[:3]



#calculates a position above the object by adding an offset to its z coordinate.
def object_approach(object_position, approach_height = 0.1): 
    approach_position = np.asarray(object_position, dtype = float).copy()
    approach_position[2] += approach_height
    grasp_object =  approach_position .copy()
    return approach_position, grasp_object