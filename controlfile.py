"""
assessment_control.py
---------------------
Assessment support module for the 41014 UR3e MuJoCo project.

Provides:
1. A simple six-joint discrete state-space model.
2. Controllability and observability rank tests.
3. A Kalman observer that estimates joint velocities from joint-position readings.
4. An LQR state-feedback calculation for a double-integrator joint model.
5. Simple tracking-error metrics for evaluation.

IMPORTANT MODEL ASSUMPTION
The joint model is an approximate double-integrator:
    q[k+1]  = q[k] + dt*dq[k] + 0.5*dt**2*u[k]
    dq[k+1] = dq[k] + dt*u[k]
where u is a virtual joint-acceleration input.

Your MuJoCo model currently uses position actuators. Therefore, this module's LQR
output is a virtual acceleration, NOT a MuJoCo motor command. Do not connect the
acceleration directly to data.ctrl. Use the model for analysis/observer work, and
only use the LQR controller in the main simulation after adding and validating
an interface suitable for your actual position actuators.

This file deliberately does not claim to calibrate the camera or prove robustness:
those require measured tests and results from your simulation.
"""

import numpy as np


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


class UR3eStateSpaceModel:
    """
    Approximate six-joint double-integrator model.

    State:
        x = [q1...q6, dq1...dq6]
        q  = joint angles (rad)
        dq = joint velocities (rad/s)

    Input:
        u = six virtual joint accelerations (rad/s^2)

    Output:
        y = six measured joint angles (rad)

    This is a teaching/analysis model, not a full nonlinear MuJoCo dynamics model.
    """

    def __init__(self, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.dt = float(dt)
        n = 6
        I = np.eye(n)
        Z = np.zeros((n, n))

        self.A = np.block([
            [I, self.dt * I],
            [Z, I],
        ])

        self.B = np.vstack([
            0.5 * self.dt**2 * I,
            self.dt * I,
        ])

        self.C = np.hstack([I, Z])
        self.D = np.zeros((6, 6))

    def analyse(self):
        """Return the controllability and observability test results."""
        return matrix_rank_report(self.A, self.B, self.C)

    def print_report(self):
        """Print a short report suitable for project testing."""
        report = self.analyse()
        print("\n--- UR3e approximate state-space model ---")
        print("State x = [joint positions, joint velocities]")
        print("Input u = virtual joint acceleration")
        print("Output y = measured joint positions")
        print(f"State matrix A shape: {self.A.shape}")
        print(f"Input matrix B shape: {self.B.shape}")
        print(f"Output matrix C shape: {self.C.shape}")
        print(
            f"Controllability rank: {report['controllability_rank']}/"
            f"{report['number_of_states']} -> {report['controllable']}"
        )
        print(
            f"Observability rank: {report['observability_rank']}/"
            f"{report['number_of_states']} -> {report['observable']}"
        )
        print("Reminder: these results apply to this approximate model.")
        return report


class JointKalmanObserver:
    """
    Estimate six joint angles and six joint velocities from joint-angle readings.

    The simulation provides qpos as the measurement. The observer estimates dq
    instead of directly using MuJoCo qvel. This makes the observer's role clear.
    """

    def __init__(self, model, position_noise=1e-5, process_noise=1e-4):
        self.A = model.A.copy()
        self.B = model.B.copy()
        self.C = model.C.copy()

        self.Q = np.eye(12) * float(process_noise)
        self.R = np.eye(6) * float(position_noise)
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

    def update(self, measured_q, acceleration_input=None):
        """
        Update the state estimate.

        acceleration_input is the previous virtual acceleration command.
        Leave it as None when no such command is available.
        """
        q_measured = np.asarray(measured_q, dtype=float).reshape(6)

        if not self.initialized:
            return self.reset(q_measured)

        if acceleration_input is None:
            u = np.zeros(6)
        else:
            u = np.asarray(acceleration_input, dtype=float).reshape(6)

        # Predict the next joint state and its uncertainty.
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Compare predicted joint angles with the measured angles.
        innovation = q_measured - self.C @ x_pred
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        # Correct the state estimate.
        self.x = x_pred + K @ innovation
        self.P = (np.eye(12) - K @ self.C) @ P_pred

        return self.x[:6].copy(), self.x[6:].copy()

    def position_standard_deviation(self):
        """Return estimated standard deviation for each joint angle (rad)."""
        return np.sqrt(np.maximum(np.diag(self.P)[:6], 0.0))


def solve_discrete_lqr(A, B, Q, R, iterations=1000, tolerance=1e-9):
    """
    Solve the discrete-time LQR Riccati equation using simple iteration.

    Returns K for u = -K x. Only NumPy is required.
    """
    P = Q.copy()

    for _ in range(iterations):
        middle = R + B.T @ P @ B
        K = np.linalg.solve(middle, B.T @ P @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q

        if np.max(np.abs(P_next - P)) < tolerance:
            P = P_next
            break

        P = P_next

    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class JointLQRController:
    """
    LQR controller for the approximate double-integrator model.

    This class calculates virtual acceleration:
        u = -K [q - q_reference, dq]

    It is included for model/controller study. It does NOT directly drive the
    existing MuJoCo position actuators. Validate a position-actuator interface
    before using it to move the arm.
    """

    def __init__(self, model, position_weight=20.0, velocity_weight=2.0,
                 input_weight=0.5):
        self.model = model

        Q = np.diag(
            [position_weight] * 6 + [velocity_weight] * 6
        )
        R = np.eye(6) * input_weight

        self.K, self.P = solve_discrete_lqr(
            model.A, model.B, Q, R
        )

    def acceleration(self, q_reference, q_estimated, dq_estimated):
        """Return the six virtual acceleration commands (rad/s^2)."""
        q_reference = np.asarray(q_reference, dtype=float).reshape(6)
        q_estimated = np.asarray(q_estimated, dtype=float).reshape(6)
        dq_estimated = np.asarray(dq_estimated, dtype=float).reshape(6)

        state_error = np.r_[
            q_estimated - q_reference,
            dq_estimated
        ]
        return -self.K @ state_error


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


if __name__ == "__main__":
    # Run this file directly to check the model and matrix ranks.
    model = UR3eStateSpaceModel(dt=1.0 / 30.0)
    model.print_report()

    observer = JointKalmanObserver(model)
    q_example = np.zeros(6)
    q_est, dq_est = observer.update(q_example)
    print("\nExample observer output:")
    print("Estimated joint positions:", np.round(q_est, 4))
    print("Estimated joint velocities:", np.round(dq_est, 4))

    controller = JointLQRController(model)
    print("LQR gain matrix shape:", controller.K.shape)
    print("This file's LQR output is virtual acceleration, not motor position.")
