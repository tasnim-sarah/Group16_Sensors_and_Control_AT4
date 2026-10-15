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
                is_joint_actuator = (model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT)
                controls_this_joint = (model.actuator_trnid[actuator_id,0] == joint_id)
                if is_joint_actuator and controls_this_joint:
                    matches.append(actuator_id)
            if len(matches) != 1:
                raise ValueError(f"Expected one exact joint actuator for {name},but found {len(matches)}")
            actuator_ids.append(matches[0])
    return np.array(actuator_ids, dtype=int)

# finds the joint angles that place the end effector at a specified position.
#how the joint angles change over time from current configuration to the target configuration

def solve_ik(robot,target_position, q_current, damping = 0.01, max_steps = 100, tolerance = 1e-6):
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
    result = robot.ikine_LM(target_pose, q0=q_current, mask=[1,1,1,0,0,0],method="chan", k = damping, ilimit = max_steps, tol = tolerance)
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


# uses the camera's instrinsic calibration paramters and depth to calculate a 3d point. world frame

def pixel_to_camerapoint(u,v,depth,fx,fy,cx,cy):
    xcamera = (u- cx) * depth/ fx
    ycamera = (v- cy) * depth / fy 
    zcamera = depth 
    return np.array([xcamera,ycamera,zcamera])

#transforms that point into the robot's base frame. 

def cameratobase(point_camera, Tbasecamera):
    homogenous_point  = np.append(point_camera, 1)
    point_base_homogenous = Tbasecamera @ homogenous_point
    return point_base_homogenous[:3]



#calculates a position above the object by adding an offset to its z coordinate.
def object_approach(object_position, approach_height = 0.1): 
    object_position = np.asarray(object_position, dtype = float)
    approach_position = object_position.copy()
    approach_position[2] += approach_height
    grasp_object =  object_position .copy()
    return approach_position, grasp_object

def make_state (dt, n_joints): 
    if dt <= 0 or n_joints < 1: 
        raise ValueError("dt and n_joints need to be positive.")
    I = np.eye(n_joints)
    Z = np.zeros((n_joints, n_joints))
    A = np.block([[I,dt * I],[Z,I]])
    B = np.vstack([0.5 * dt **2 * I,dt* I])
    C = np.hstack([I,Z])
    D = np.zeros((n_joints, n_joints))
    return A, B, C, D

def controllability_matrix (A, B):
    n_states = A.shape[0]
    blocks = []
    current = B.copy()
    for _ in range (n_states): 
        blocks.append(current)
        current = A @ current
    return np.hstack(blocks)

def observability_matrix(A,C):
    n_states = A.shape[0]
    blocks = []
    current = C.copy()
    for _ in range(n_states):
        blocks.append(current)
        current = current @ A
    return np.vstack(blocks)

def analyse_system(A,B,C):
    n_states = A.shape[0]
    controllability_rank = np.linalg.matrix_rank(controllability_matrix(A,B))
    observability_rank = np.linalg.matrix_rank(observability_matrix(A, C))
    print("number of states:", n_states)
    print("Controllability rank:", controllability_rank)
    print("fully controllable: ", controllability_rank == n_states)
    print("Observability rank:", observability_rank)
    print("Fully observable:", observability_rank == n_states)
    return controllability_rank, observability_rank

def kalman_predict(xhat, P,A, Q):
    xpred = A @ xhat
    Ppred = A @ P @ A.T + Q 
    return xpred, Ppred

def update_kalman(xpred, Ppred, y,C, R):
    diff = y - C @ xpred
    S = C @ Ppred @ C.T + R
    K = np.linalg.solve(S, C @ Ppred).T
    xhat = xpred + K @ diff
    I = np.eye(Ppred.shape[0])
    correction = I - K @ C
    P = correction @ Ppred @ correction.T + K @ R @ K.T
    return xhat, P

def feedbackController( qref, qd_ref, qhat, qd_hat, kp = 0.2, kd = 0.02):
    pos_error = qref - qhat
    vel_error = qd_ref - qd_hat
    q_command = (qref + kp * pos_error + kd * vel_error)
    return q_command



def execute_feedback_trajectory(model,data,actuator_ids,qpos_addresses,times,trajectory,kp=0.2,kd=0.02,rng=None):
    # Create a random-number generator for simulated measurement noise.
    if rng is None:
        rng = np.random.default_rng()

    # Read MuJoCo's actual physics timestep.
    dt = model.opt.timestep

    # Determine the number of controlled joints.
    n_joints = trajectory.shape[1]
    n_states = 2 * n_joints


    I = np.eye(n_joints)
    Z = np.zeros((n_joints, n_joints))

    # State = [joint positions, joint velocities].
    A = np.block([
        [I, dt * I],
        [Z, I]
    ])

    # We measure positions only.
    C = np.hstack([I, Z])

    # Example noise assumptions. These need tuning.
    Q = np.diag(
        [1e-7] * n_joints + [1e-4] * n_joints
    )

    measurement_std = 0.002
    R = np.eye(n_joints) * measurement_std**2



    q_initial = data.qpos[qpos_addresses].copy()

    # Start by assuming the joints are stationary.
    qd_initial = np.zeros(n_joints)

    xhat = np.concatenate([q_initial, qd_initial])
    P = np.eye(n_states) * 0.01

    # Calculate desired joint velocities from the trajectory.
    qd_trajectory = np.gradient(
        trajectory, times, axis=0
    )

    # Number of MuJoCo simulation steps.
    total_steps = int(np.ceil(times[-1] / dt))

   

    for step in range(total_steps + 1):

        # Current time, limited to the trajectory duration.
        t = min(step * dt, times[-1])

        # Get the desired position for every joint.
        qref = np.array([
            np.interp(t, times, trajectory[:, j])
            for j in range(n_joints)
        ])

        # Get the desired velocity for every joint.
        qd_ref = np.array([
            np.interp(t, times, qd_trajectory[:, j])
            for j in range(n_joints)
        ])

    

        q_measured = data.qpos[qpos_addresses].copy()

        # Add artificial noise to demonstrate filtering.
        # For a real sensor, replace this with its measurements.
        y = q_measured + rng.normal(
            0.0, measurement_std, size=n_joints
        )

     

        # Predict the next state.
        xpred, Ppred = kalman_predict(
            xhat, P, A, Q
        )

        # Correct the prediction using measured positions.
        xhat, P = update_kalman(
            xpred, Ppred, y, C, R
        )

        # Extract estimated positions and velocities.
        qhat = xhat[:n_joints]
        qd_hat = xhat[n_joints:]

     

        q_command = feedbackController(
            qref,
            qd_ref,
            qhat,
            qd_hat,
            kp=kp,
            kd=kd
        )


        data.ctrl[actuator_ids] = q_command

    

        if step < total_steps:
            mujoco.mj_step(model, data)
