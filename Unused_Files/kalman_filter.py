
# ============================================================
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

def analyse_system(A,B, C):
    controllability = controllability_matrix(A,B)
    observability = observability_matrix(A, C)
    n_states = A.shape[0]
    print("number of states:", n_states)
    print("Controllability rank:", controllability_rank)
    print("fully controllable: ", controllability_rank == n_states)
    print("Observability rank:", observability_rank)
    print("Fully observable:", observability_rank == n_states)
    return contrability_rank, observability_rank

def kalman_predict(xhat, P,A, Q):
    xpred = A @ xhat
    Ppred = A @ P @ A.T + Q 
    return xpred, Ppred

def update_kalman(xpred, Ppred, y,C, R):
    diff = y - C @ xpred
    S = C @ Ppred @ C.T + R
    K = np.linalg.solve(S.T, Ppred @ C.T).T)
    xhat = xpred + K @ diff
    I = np.eye(ppred.shape[0])
    correction = I - K @ C
    P = correction @ Ppred @ correction.T + K @ R @ K.T
    return xhat, P

def feedbackController( qref, qd_ref, qhat, qd_hat, kp = 0.2, kd = 0.02):
    pos_error = qref - qhat
    vel_error = qd_ref - qd_hat
    q_command = (qref + kp * pos_error + kd * vel_error)
    return q_command

