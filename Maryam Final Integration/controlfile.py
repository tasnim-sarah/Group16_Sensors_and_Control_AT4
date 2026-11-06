"""State-space models, estimation and LQR control for the UR3e MuJoCo simulation."""

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are


def controllability_matrix(A, B):
    """Return [B, A B, A^2 B, ... A^(n-1) B]."""
    n = A.shape[0]
    blocks = []
    power = np.eye(n)
    for _ in range(n):
        blocks.append(power @ B)
        power = power @ A
    return np.hstack(blocks)


def observability_matrix(A, C):
    """Return [C; C A; C A^2; ... C A^(n-1)]."""
    n = A.shape[0]
    blocks = []
    power = np.eye(n)
    for _ in range(n):
        blocks.append(C @ power)
        power = power @ A
    return np.vstack(blocks)


def matrix_rank_report(A, B, C, tolerance=None):
    """Calculate controllability and observability ranks and full-rank flags."""
    ctrb = controllability_matrix(A, B)
    obsv = observability_matrix(A, C)
    n = A.shape[0]

    rank_ctrb = int(np.linalg.matrix_rank(ctrb, tol=tolerance))
    rank_obsv = int(np.linalg.matrix_rank(obsv, tol=tolerance))

    return {
        "controllability_matrix": ctrb,
        "observability_matrix": obsv,
        "controllability_rank": rank_ctrb,
        "observability_rank": rank_obsv,
        "number_of_states": n,
        "controllable": rank_ctrb == n,
        "observable": rank_obsv == n,
    }

class PositionActuatorStateSpaceModel:
    """Local discrete model of the six MuJoCo UR3e position-servo actuators.

    MuJoCo's transition finite differences are ordered as joint-position
    perturbations followed by joint-velocity perturbations. The resulting input
    is the actuator's target joint position, not torque or acceleration.
    """

    def __init__(self, model, data, finite_difference_step=1e-6):
        if model.nv < 6 or model.nu < 6:
            raise ValueError("the model must contain six UR3e joints and actuators")

        for index in range(6):
            if model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("the first six actuators must be joint actuators")
            if int(model.actuator_trnid[index, 0]) != index:
                raise ValueError("the first six actuators must control UR3e joints in order")
            if model.actuator_biastype[index] != mujoco.mjtBias.mjBIAS_AFFINE:
                raise ValueError("the UR3e joint actuators must use affine position-servo bias")

        self.dt = float(model.opt.timestep)
        self.nominal_state = np.r_[data.qpos[:6], data.qvel[:6]].astype(float)
        self.nominal_input = data.ctrl[:6].astype(float).copy()
        self.ctrl_limits = model.actuator_ctrlrange[:6].astype(float).copy()

        nominal_data = mujoco.MjData(model)
        mujoco.mj_copyData(nominal_data, model, data)
        state_size = 2 * model.nv + model.na
        full_a = np.zeros((state_size, state_size), dtype=float)
        full_b = np.zeros((state_size, model.nu), dtype=float)
        mujoco.mjd_transitionFD(
            model, nominal_data, finite_difference_step, True,
            full_a, full_b, None, None
        )

        velocity_start = model.nv
        self.A = np.block([
            [full_a[:6, :6], full_a[:6, velocity_start:velocity_start + 6]],
            [full_a[velocity_start:velocity_start + 6, :6],
             full_a[velocity_start:velocity_start + 6, velocity_start:velocity_start + 6]],
        ])
        self.B = np.vstack([
            full_b[:6, :6],
            full_b[velocity_start:velocity_start + 6, :6],
        ])
        self.C = np.hstack([np.eye(6), np.zeros((6, 6))])
        self.D = np.zeros((6, 6))

        next_data = mujoco.MjData(model)
        mujoco.mj_copyData(next_data, model, data)
        mujoco.mj_step(model, next_data)
        self.nominal_next_state = np.r_[
            next_data.qpos[:6], next_data.qvel[:6]
        ].astype(float)
        self.offset = (
            self.nominal_next_state
            - self.A @ self.nominal_state
            - self.B @ self.nominal_input
        )
        self.validation_max_error = self._validate_local_model(model, data)

    def analyse(self):
        """Return controllability and observability ranks for this local model."""
        return matrix_rank_report(self.A, self.B, self.C)

    def print_report(self):
        """Print the model's actuator interface and local validation results."""
        report = self.analyse()
        print("\n--- UR3e MuJoCo position-actuator state-space model ---")
        print("State x = [six joint positions, six joint velocities]")
        print("Input u = six joint-position actuator targets (rad)")
        print(f"MuJoCo timestep: {self.dt:.6f} s")
        print(
            f"Controllability rank: {report['controllability_rank']}/12; "
            f"observability rank: {report['observability_rank']}/12"
        )
        print(
            "One-step local-model max residual: "
            f"{self.validation_max_error:.3e} (small perturbation test)"
        )
        return report

    def _validate_local_model(self, model, data):
        """Compare a small perturbed MuJoCo step with the linear prediction."""
        rng = np.random.default_rng(0)
        delta_state = np.r_[
            rng.uniform(-1e-3, 1e-3, 6),
            rng.uniform(-1e-2, 1e-2, 6),
        ]
        delta_input = rng.uniform(-1e-3, 1e-3, 6)

        perturbed_data = mujoco.MjData(model)
        mujoco.mj_copyData(perturbed_data, model, data)
        perturbed_data.qpos[:6] += delta_state[:6]
        perturbed_data.qvel[:6] += delta_state[6:]
        perturbed_data.ctrl[:6] += delta_input
        mujoco.mj_forward(model, perturbed_data)
        mujoco.mj_step(model, perturbed_data)
        actual = np.r_[
            perturbed_data.qpos[:6], perturbed_data.qvel[:6]
        ]
        predicted = (
            self.nominal_next_state + self.A @ delta_state + self.B @ delta_input
        )
        return float(np.max(np.abs(actual - predicted)))


class JointKalmanObserver:
    """
    Estimate six joint angles and six joint velocities from joint-angle readings.

    The simulation provides qpos as the measurement. The observer estimates dq
    instead of directly using MuJoCo qvel. This makes the observer's role clear.
    """

    def __init__(self, model, position_noise=2.5e-7, process_noise=1e-5):
        self.A = model.A.copy()
        self.B = model.B.copy()
        self.C = model.C.copy()
        self.offset = model.offset.copy()

        self.Q = np.eye(12) * float(process_noise)
        self.R = np.eye(6) * max(float(position_noise), 1e-12)
        self.P = np.eye(12)
        self.x = np.zeros(12)
        self.initialized = False

    def reset(self, measured_q):
        """Start the observer from the first six-joint measurement."""
        q = np.asarray(measured_q, dtype=float).reshape(6)
        self.x[:] = 0.0
        self.x[:6] = q
        self.P = np.eye(12)
        self.initialized = True
        return self.x[:6].copy(), self.x[6:].copy()

    def update(self, measured_q, control_input=None):
        """
        Update the state estimate.

        control_input is the previous six position-actuator target commands.
        """
        q_measured = np.asarray(measured_q, dtype=float).reshape(6)

        if not self.initialized:
            return self.reset(q_measured)

        if control_input is None:
            u = np.zeros(6)
        else:
            u = np.asarray(control_input, dtype=float).reshape(6)

        # Predict the next joint state and its uncertainty.
        x_pred = self.A @ self.x + self.B @ u + self.offset
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Compare predicted joint angles with the measured angles.
        innovation = q_measured - self.C @ x_pred
        S = self.C @ P_pred @ self.C.T + self.R
        K = np.linalg.solve(S, self.C @ P_pred).T
        self.last_nis = float(innovation @ np.linalg.solve(S, innovation))

        # Correct the state estimate.
        self.x = x_pred + K @ innovation
        residual_map = np.eye(12) - K @ self.C
        self.P = residual_map @ P_pred @ residual_map.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) * 0.5

        return self.x[:6].copy(), self.x[6:].copy()

    def position_standard_deviation(self):
        """Return estimated standard deviation for each joint angle (rad)."""
        return np.sqrt(np.maximum(np.diag(self.P)[:6], 0.0))


def solve_discrete_lqr(A, B, Q, R, iterations=5000, tolerance=1e-9):
    """
    Solve the discrete-time LQR Riccati equation with a Schur solver.

    Returns K for u = -K x; refuses an unstable local closed loop.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    if np.max(np.abs(np.linalg.eigvals(A - B @ K))) >= 1:
        raise RuntimeError("LQR design is not stable for the local model")
    return K, P



class PositionActuatorLQRController:
    """LQR feedback with a model-derived steady-state position target."""

    def __init__(self, model, position_weight=20.0, velocity_weight=1.0,
                 input_weight=0.1):
        self.model = model
        Q = np.diag([position_weight] * 6 + [velocity_weight] * 6)
        R = np.eye(6) * input_weight
        self.Q, self.R = Q.copy(), R.copy()
        self.K, self.P = solve_discrete_lqr(model.A, model.B, Q, R)
        input_pseudoinverse = np.linalg.pinv(model.B)
        self._reference_map = (
            input_pseudoinverse @ (np.eye(12) - model.A)[:, :6]
        )
        self._reference_offset = -input_pseudoinverse @ model.offset

    def raw_command(self, q_reference, q_estimated, dq_estimated):
        """Return the unconstrained actuator target from estimated state feedback."""
        q_reference = np.asarray(q_reference, dtype=float).reshape(6)
        q_estimated = np.asarray(q_estimated, dtype=float).reshape(6)
        dq_estimated = np.asarray(dq_estimated, dtype=float).reshape(6)
        feedforward = self._reference_map @ q_reference + self._reference_offset
        error = np.r_[q_estimated - q_reference, dq_estimated]
        return feedforward - self.K @ error

    def command(self, q_reference, q_estimated, dq_estimated):
        """Return six clipped position targets with steady-state feedforward."""
        return np.clip(
            self.raw_command(q_reference, q_estimated, dq_estimated),
            self.model.ctrl_limits[:, 0],
            self.model.ctrl_limits[:, 1],
        )

def tracking_metrics(reference, measured):
    """
    Calculate joint-position RMSE and maximum absolute error.

    reference and measured should be arrays shaped (samples, 6), in radians.
    """
    reference = np.asarray(reference, dtype=float)
    measured = np.asarray(measured, dtype=float)

    if reference.shape != measured.shape:
        raise ValueError("reference and measured arrays must have the same shape")
    if reference.ndim != 2 or reference.shape[1] != 6:
        raise ValueError("expected arrays with shape (samples, 6)")
    if reference.shape[0] == 0:
        raise ValueError("at least one sample is required")

    error = measured - reference
    return {
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "rmse_per_joint_rad": np.sqrt(np.mean(error**2, axis=0)),
    }


if __name__ == '__main__':
    from robot_scene import build_scene
    simulation, initial = build_scene(7)
    dynamics = PositionActuatorStateSpaceModel(simulation, initial)
    dynamics.print_report()
    feedback = PositionActuatorLQRController(dynamics)
    print('Closed-loop spectral radius:', np.max(abs(np.linalg.eigvals(dynamics.A-dynamics.B@feedback.K))))
    print('Run exampletestfile.py to execute sorting with estimated-state LQR feedback.')
