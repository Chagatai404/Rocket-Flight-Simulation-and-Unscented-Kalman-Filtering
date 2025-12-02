"""UKF for 3D rocket state estimation.

This module contains a lightweight Unscented Kalman Filter that estimates
[position, velocity, attitude quaternion, angular rates]. The filter uses
IMU accelerometer readings as *inputs* to the process model and updates
altitude using the barometric measurement (converted to altitude upstream).

Key design choices (as patched during debugging):
  • Simulator now produces IMU *specific force* (a − g) in BODY.
  • Process model rotates BODY→WORLD and ADDS gravity before integrating.
  • Accel/gyro are not used as measurements; altitude-only update avoids
    non-linear pressure coupling that previously bent the estimate.
  • Measurement model is scalar altitude: z = state[2].
  • Integration uses semi‑implicit Euler: pos += vel_new * dt.
  • Helper methods provide UKF‑aided smoothing of IMU for plots.
"""
import numpy as np
from collections import deque

class UKF:
    """Minimal 13‑state UKF: [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz].

    Notes
    -----
    * Accel (specific force) is treated as a known input (from IMU)
      and used only in the prediction step. We *add back gravity* after
      rotating into the world frame so that integrated acceleration
      equals physical linear acceleration.
    * Measurement update is scalar altitude (meters) for numerical
      robustness and tight altitude matching.
    """
    def __init__(self, dt):
        """
        Initialize the Unscented Kalman Filter with tuning parameters, 
        initial state, and noise covariances.

        Parameters:
            dt (float): Time step in seconds.
        """
        self.dt = dt
        self.n = 13  # 3 pos, 3 vel, 4 quat, 3 omega

        # --- UKF tuning parameters ---
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        # --- Sigma point weights ---
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)

        # --- Initial state and covariance ---
        self.x = np.zeros(self.n)
        self.x[6] = 1.0  # identity quaternion (qw, qx, qy, qz) with qw in index 6
        self.P = np.eye(self.n) * 0.1

        # --- Process noise covariance ---
        self.Q = np.diag([
            0.01, 0.01, 0.01,      # position
            0.1, 0.1, 0.1,         # velocity
            0.001, 0.001, 0.001, 0.001,  # quaternion
            0.05, 0.05, 0.05       # angular velocity
        ])

        # --- Measurement noise covariance (altitude only) ---
        self.R = np.array([[ (8.33*5.0)**2 ]])       # barometric pressure noise

        # --- Barometer model (unchanged; keep altitude estimation intact) ---
        self.P0 = 1013.25  # hPa
        self.H = 8434.5    # m

        # --- Diagnostics / filtered sensor outputs ---
        # Latest body-frame acceleration used in prediction (net, as provided to predict step)
        self.latest_body_acceleration = np.zeros(3)

        # UKF-aided, denoised sensor streams for plotting (EMA over a UKF/model blend)
        self._accel_ema = np.zeros(3)  # internal EMA state for accel
        self._gyro_ema  = np.zeros(3)  # internal EMA state for gyro
        self.filtered_accel_body = np.zeros(3)  # published filtered accel (body frame)
        self.filtered_gyro_body  = np.zeros(3)  # published filtered gyro (body frame)

        # EMA parameters for sensor smoothing (feel free to tune)
        self.accel_ema_alpha = 0.15     # smaller -> smoother accel
        self.gyro_ema_alpha  = 0.20     # smaller -> smoother gyro

        # Blend weight parameter: how much we trust measurement vs model (gravity-only)
        # We compute a dynamic weight each step; this is the fallback/ceiling.
        self.max_model_blend = 0.7  # at most 70% gravity model when we're very static

        # --- Minimal accel spike suppressor (rolling median + soft clamp) ---
        self.acc_spike_window = 5          # samples for rolling median (short)
        self.acc_soft_jump_max = 6.0       # max per-sample change after median (m/s^2)
        # Gate to avoid suppressing true thrust along Z when far from 1g
        self.acc_thrust_gate_dev = 2.0    # if | |a| - 9.81 | > gate -> relax Z clamp
        self.acc_soft_jump_max_z_thrust = float('inf')  # disable Z clamp during thrust
        self._acc_window = deque(maxlen=self.acc_spike_window)

    def generate_sigma_points(self):
        """
        Generate 2n+1 sigma points around the current state estimate.

        Returns:
            sigma_points (np.ndarray): Array of shape (2n+1, n) of sigma points.
        """
        n = self.n
        sigma_points = np.zeros((2 * n + 1, n))
        scaled_P = (self.n + self.lambda_) * self.P

        # Enforce symmetry to reduce numerical instability
        scaled_P = 0.5 * (scaled_P + scaled_P.T)

        # Robust Cholesky with jitter
        for eps in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
            try:
                sqrt_matrix = np.linalg.cholesky(scaled_P + np.eye(n) * eps)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            # As a last resort: clamp eigenvalues and retry
            eigvals, eigvecs = np.linalg.eigh(scaled_P)
            eigvals_clamped = np.clip(eigvals, 1e-8, None)
            scaled_P = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
            sqrt_matrix = np.linalg.cholesky(scaled_P)

        # Mean
        sigma_points[0] = self.x.flatten()

        # +/- columns of sqrt_matrix
        for i in range(n):
            sigma_points[i + 1]     = self.x + sqrt_matrix[:, i]
            sigma_points[i + 1 + n] = self.x - sqrt_matrix[:, i]

        return sigma_points

    def predict_sigma_points(self, sigma_points, accel_measured):
        """
        Propagate sigma points through the process model using measured body-frame acceleration.

        Parameters:
            sigma_points (np.ndarray): Sigma points of shape (2n+1, n).
            accel_measured (np.ndarray): Acceleration vector in BODY frame (3,),
                                         i.e., what the IMU reports after bias/noise.

        Returns:
            predicted (np.ndarray): Predicted sigma points (2n+1, n).
        """
        dt = self.dt
        predicted = np.zeros_like(sigma_points)

        for i, x in enumerate(sigma_points):
            # --- Extract state ---
            pos  = x[0:3]
            vel  = x[3:6]
            quat = x[6:10]
            omega = x[10:13]

            # --- Rotate measured specific force to WORLD and add gravity ---
            # IMU reports specific force f = a - g in BODY.
            # Rotate to WORLD then add gravity to recover linear acceleration a.
            # rotate_vector performs BODY->WORLD for a vector expressed in BODY.
            f_world = self.rotate_vector(quat, accel_measured)
            a_world = f_world + np.array([0.0, 0.0, -9.81])

            # --- Kinematics integration (semi-implicit Euler) ---
            vel_new = vel + a_world * dt
            pos_new = pos + vel * dt

            # --- Quaternion integration using angular velocity ---
            omega_quat = np.zeros(4)
            omega_quat[1:] = omega
            dq = 0.5 * self.quaternion_multiply(quat, omega_quat) * dt
            quat_new = quat + dq
            quat_new /= np.linalg.norm(quat_new)  # normalize to prevent drift

            # --- Assemble predicted sigma point ---
            x_pred = np.zeros(self.n)
            x_pred[0:3]  = pos_new
            x_pred[3:6]  = vel_new
            x_pred[6:10] = quat_new
            x_pred[10:13] = omega
            predicted[i] = x_pred

        # For diagnostics/output (what we used to advance the state)
        self.latest_body_acceleration = accel_measured.copy()

        return predicted

    @staticmethod
    def quaternion_multiply(q, r):
        """
        Multiply two quaternions.

        Parameters:
            q (np.ndarray): First quaternion (4,).
            r (np.ndarray): Second quaternion (4,).

        Returns:
            np.ndarray: Resulting quaternion (4,).
        """
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ])

    def predict_mean_and_covariance(self, sigma_points_pred):
        """
        Compute the predicted state mean and covariance from predicted sigma points.

        Parameters:
            sigma_points_pred (np.ndarray): Predicted sigma points (2n+1, n).

        Returns:
            None. Updates self.x and self.P in place.
        """
        # --- Weighted mean ---
        x_pred = np.zeros(self.n)
        for i in range(sigma_points_pred.shape[0]):
            x_pred += self.Wm[i] * sigma_points_pred[i]

        # --- Normalize quaternion part of the mean ---
        quat_mean = x_pred[6:10].copy()
        x_pred[6:10] = quat_mean / np.linalg.norm(quat_mean)

        # --- Covariance from deviations ---
        P_pred = np.zeros((self.n, self.n))
        for i in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        # Add process noise
        P_pred += self.Q

        # Commit
        self.x = x_pred
        self.P = P_pred

    def predict_measurement(self, sigma_points_pred):
        """Predict altitude measurement from predicted sigma points.
        Returns:
            z_pred (np.ndarray): (1,) predicted altitude mean
            S (np.ndarray): (1,1) measurement covariance
            Z_sigma (np.ndarray): (2n+1,1) sigma-point measurements
        """
        n_sigma = sigma_points_pred.shape[0]
        Z_sigma = np.zeros((n_sigma, 1))

        for i, sigma in enumerate(sigma_points_pred):
            # altitude is world-frame z position at index 2
            altitude_pred = sigma[2]
            Z_sigma[i, 0] = altitude_pred

        # Predicted mean measurement
        z_pred = np.dot(self.Wm, Z_sigma).reshape(1)

        # Measurement covariance
        S = np.zeros((1, 1))
        for i in range(n_sigma):
            dz = Z_sigma[i] - z_pred
            S += self.Wc[i] * (dz @ dz.T)
        S += self.R
        return z_pred, S, Z_sigma

    @staticmethod
    def rotate_vector(q, v):
        """
        Rotate a vector v from BODY frame to WORLD frame using quaternion q.

        Parameters:
            q (np.ndarray): Quaternion (4,) with layout [qw, qx, qy, qz].
            v (np.ndarray): Vector in BODY frame (3,).

        Returns:
            np.ndarray: Rotated vector in WORLD frame (3,).
        """
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q
        q_conj = np.array([qw, -qx, -qy, -qz])
        v_quat = np.array([0.0, *v])
        rotated = UKF.quaternion_multiply(
            UKF.quaternion_multiply(q, v_quat), q_conj
        )
        return rotated[1:]

    def _update_sensor_filters(self, accel_meas_body, gyro_meas_body):
        """
        Internal helper: form a UKF-aided, denoised accel/gyro stream.

        We blend the gravity-only model (from attitude) with the raw measurement.
        Weighting is dynamic:
        - If |accel| ≈ 1g, we prefer the model (gravity dominates; measurement can be noisy).
        - If |accel| deviates from 1g (thrust/maneuvers), we prefer the measurement.

        Then we apply an EMA to suppress spikes.

        Parameters:
            accel_meas_body (np.ndarray): Measured body-frame acceleration (3,).
            gyro_meas_body (np.ndarray):  Measured body-frame angular rate (3,).

        Returns:
            None. Updates self.filtered_accel_body and self.filtered_gyro_body.
        """
        # --- Accel prefilter: rolling median + soft clamp (spike removal) ---
        self._acc_window.append(np.asarray(accel_meas_body, dtype=float))
        if len(self._acc_window) > 0:
            acc_med = np.median(np.stack(self._acc_window, axis=0), axis=0)
        else:
            acc_med = np.asarray(accel_meas_body, dtype=float)

        # Soft clamp relative to previous published accel (prevents single-sample spikes)
        prev = self.filtered_accel_body if np.any(self.filtered_accel_body) else acc_med
        delta = acc_med - prev

        # Compute deviation from 1g to detect high-thrust phases
        acc_mag_tmp = float(np.linalg.norm(acc_med))
        dev_from_1g_tmp = abs(acc_mag_tmp - 9.81)

        # Per-axis clamp: X/Y always clamped; Z relaxed when strong thrust detected
        clamp_xy = self.acc_soft_jump_max
        clamp_z  = self.acc_soft_jump_max if dev_from_1g_tmp <= self.acc_thrust_gate_dev else self.acc_soft_jump_max_z_thrust
        delta[0] = np.clip(delta[0], -clamp_xy, clamp_xy)
        delta[1] = np.clip(delta[1], -clamp_xy, clamp_xy)
        delta[2] = np.clip(delta[2], -clamp_z,  clamp_z)

        accel_prefiltered = prev + delta

        # --- Gravity model in BODY frame from current attitude ---
        g_world = np.array([0, 0, -9.81])
        g_body  = self.rotate_vector(self.x[6:10], g_world)
        gravity_only = -g_body  # what a perfect stationary accelerometer would read

        # --- Dynamic blend weight based on how "non-1g" we are ---
        accel_mag = np.linalg.norm(accel_prefiltered)
        dev_from_1g = abs(accel_mag - 9.81)                   # how far from 1g
        # Map deviation to [0..1]; 0 -> fully model, big deviation -> fully measurement
        k = 4.0  # [m/s^2] deviation for ~100% measurement; tune as needed
        meas_w = np.clip(dev_from_1g / k, 0.0, 1.0)
        # Cap model contribution so we never over-trust it
        model_w = np.minimum(1.0 - meas_w, self.max_model_blend)

        # --- Blended accel before EMA ---
        blended_accel = model_w * gravity_only + (1.0 - model_w) * accel_prefiltered

        # --- EMA smoothing (accel) ---
        a = self.accel_ema_alpha
        self._accel_ema = a * self._accel_ema + (1.0 - a) * blended_accel
        self.filtered_accel_body = self._accel_ema.copy()

        # --- EMA smoothing (gyro) ---
        g = self.gyro_ema_alpha
        self._gyro_ema = g * self._gyro_ema + (1.0 - g) * gyro_meas_body
        self.filtered_gyro_body = self._gyro_ema.copy()

    def update(self, x_pred, P_pred, sigma_points_pred, z_pred, S, Z_sigma, z_measured):
        """Kalman update with generic-dim measurement z_measured."""
        n_z = z_measured.shape[0]
        P_xz = np.zeros((self.n, n_z))

        # Cross covariance
        for i in range(sigma_points_pred.shape[0]):
            dx = sigma_points_pred[i] - x_pred
            dz = Z_sigma[i] - z_pred
            P_xz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = P_xz @ np.linalg.inv(S)

        # Innovation
        dz = z_measured - z_pred

        # State & covariance update
        self.x = x_pred + K @ dz
        self.P = P_pred - K @ S @ K.T

        # Optional sensor filtering when full IMU is provided
        if n_z >= 6:
            accel_meas = z_measured[0:3]
            gyro_meas  = z_measured[3:6]
            self._update_sensor_filters(accel_meas, gyro_meas)

    # ---------------- Public accessors for plotting ----------------

    def estimate_net_body_acceleration(self):
        """
        Return latest body-frame acceleration used during the prediction step.
        (This is the *input* accel you provided to the process model.)

        Returns:
            np.ndarray: Latest body-frame acceleration (3,).
        """
        return self.latest_body_acceleration

    def get_filtered_accel(self):
        """
        Get the UKF-aided, denoised accelerometer output in the BODY frame.

        Returns:
            np.ndarray: Filtered accel (3,).
        """
        return self.filtered_accel_body.copy()

    def get_filtered_gyro(self):
        """
        Get the UKF-aided, denoised gyroscope output in the BODY frame.

        Returns:
            np.ndarray: Filtered gyro (3,).
        """
        return self.filtered_gyro_body.copy()

