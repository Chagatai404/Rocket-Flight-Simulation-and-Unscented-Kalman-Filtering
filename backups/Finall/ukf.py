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
        self.n = 13
        
        # UKF parameters (unchanged)
        self.alpha = 1e-2
        self.beta = 2
        self.kappa = 3 - self.n
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weight calculation (unchanged)
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        # --- IMPROVED: Tighter process noise for better altitude tracking ---
        self.Q_ascent = np.diag([
            0.0005, 0.0005, 0.0002,   # Tighter position, especially Z
            0.005, 0.005, 0.002,      # Tighter velocity  
            0.0001, 0.0001, 0.0001, 0.0001,
            0.001, 0.001, 0.001
        ])
        
        self.Q_descent = np.diag([
            0.01, 0.01, 0.01,      # higher position uncertainty during parachute
            0.05, 0.05, 0.05,      # higher velocity uncertainty
            0.001, 0.001, 0.001, 0.001,  # higher attitude uncertainty
            0.01, 0.01, 0.01       # higher angular rate uncertainty
        ])
        
        self.Q = self.Q_ascent.copy()
        
        # Initial state and covariance (unchanged)
        self.x = np.zeros(self.n)
        self.x[6] = 1.0
        
        self.P = np.diag([
            1.0, 1.0, 1.0,
            0.5, 0.5, 0.5,
            0.01, 0.01, 0.01, 0.01,
            0.1, 0.1, 0.1
        ])
        
        # Measurement noise (slightly increased for robustness)
        self.R = np.diag([[1.5**2, 2.0**2]])  # [z, vz]
        
        # Add flight phase detection
        self.flight_phase = "ascent"  # "ascent", "coast", "descent"
        
        # --- Apogee detection for parachute deployment ---
        self.apogee_detected = False
        self.apogee_time = None
        self.velocity_history = []
        self.altitude_history = []
        self.velocity_window = 10  # samples to look back for apogee detection

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

        # --- Velocity EMA + short history (for central-diff derivative) ---
        self.vel_ema_alpha = 0.25
        self._vel_ema = np.zeros(3)
        self._vel_hist = deque(maxlen=3)
        # Blend weight parameter: how much we trust measurement vs model (gravity-only)
        # We compute a dynamic weight each step; this is the fallback/ceiling.
        self.max_model_blend = 0.7  # at most 70% gravity model when we're very static

        # --- Minimal accel spike suppressor (rolling median + soft clamp) ---
        self.acc_spike_window = 5          # samples for rolling median (short)
        self.acc_soft_jump_max = 6.0       # max per-sample change after median (m/s^2)
        # Gate to avoid suppressing true thrust along Z when far from 1g
        self.acc_thrust_gate_dev = 3.5     # if | |a| - 9.81 | > gate -> relax Z clamp
        self.acc_soft_jump_max_z_thrust = 12.0
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

    def predict_sigma_points(self, sigma_points, accel_measured, gyro_measured=None):
        """
        Propagate sigma points through the process model using measured body-frame acceleration.

        Parameters:
            sigma_points (np.ndarray): Sigma points of shape (2n+1, n).
            accel_measured (np.ndarray): Acceleration vector in BODY frame (3,),
                                         i.e., what the IMU reports after bias/noise.
            gyro_measured (np.ndarray): Gyroscope vector in BODY frame (3,), optional.

        Returns:
            predicted (np.ndarray): Predicted sigma points (2n+1, n).
        """
        dt = self.dt
        predicted = np.zeros_like(sigma_points)

        for i, x in enumerate(sigma_points):
            pos  = x[0:3]
            vel  = x[3:6]
            quat = x[6:10]
            omega = x[10:13]

            omega_used = gyro_measured if gyro_measured is not None else omega

            # --- IMPROVED: Better thrust phase detection ---
            accel_mag = np.linalg.norm(accel_measured)
            dev_from_1g = abs(accel_mag - 9.81)
            
            if self.apogee_detected:
                # Parachute: use blended measurement
                gravity_world = np.array([0.0, 0.0, -9.81])
                g_body = self.rotate_vector(self.quaternion_conjugate(quat), gravity_world)
                parachute_blend = 0.2  # 80% gravity model, 20% measurement
                accel_used = parachute_blend * accel_measured + (1 - parachute_blend) * (-g_body)
                f_world = self.rotate_vector(quat, accel_used)
                
            elif dev_from_1g > 10.0:  # High thrust phase
                # During high thrust, use raw measurement more (less filtering)
                # This preserves rapid acceleration changes
                f_world = self.rotate_vector(quat, accel_measured)
                
            else:  # Normal operation
                # Use the pre-filtered acceleration for normal phases
                f_world = self.rotate_vector(quat, self.filtered_accel_body)
            
            a_world = f_world + np.array([0.0, 0.0, -9.81])

            # --- IMPROVED: Semi-implicit Euler with better numerical stability ---
            # First update velocity, then position with new velocity
            vel_new = vel + a_world * dt
            
            # Clamp velocity changes during parachute to prevent oscillations
            if self.apogee_detected:
                max_vel_change = 5.0 * dt  # m/s per step
                vel_change = vel_new - vel
                vel_change_mag = np.linalg.norm(vel_change)
                if vel_change_mag > max_vel_change:
                    vel_change = vel_change * (max_vel_change / vel_change_mag)
                    vel_new = vel + vel_change
            
            pos_new = pos + vel_new * dt

            # Quaternion integration (unchanged but kept for completeness)
            omega_magnitude = np.linalg.norm(omega_used)
            if omega_magnitude > 1e-8:
                omega_norm = omega_used / omega_magnitude
                angle = omega_magnitude * dt
                
                cos_half_angle = np.cos(angle / 2)
                sin_half_angle = np.sin(angle / 2)
                
                dq = np.array([
                    cos_half_angle,
                    sin_half_angle * omega_norm[0],
                    sin_half_angle * omega_norm[1],
                    sin_half_angle * omega_norm[2]
                ])
                
                quat_new = self.quaternion_multiply(dq, quat)
            else:
                quat_new = quat.copy()
            
            quat_new /= np.linalg.norm(quat_new)

            # Angular velocity handling
            if self.apogee_detected:
                alpha_gyro = 0.8  # Stronger filtering during parachute
                omega_new = alpha_gyro * omega + (1 - alpha_gyro) * omega_used
            else:
                omega_new = omega_used.copy()

            # Assemble predicted sigma point
            x_pred = np.zeros(self.n)
            x_pred[0:3]  = pos_new
            x_pred[3:6]  = vel_new
            x_pred[6:10] = quat_new
            x_pred[10:13] = omega_new
            predicted[i] = x_pred

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
        # --- Weighted mean for all states (simplified approach) ---
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

    def _quaternion_mean(self, quaternions, weights):
        """
        Compute weighted mean of quaternions using spherical averaging.
        
        Parameters:
            quaternions (np.ndarray): Array of quaternions (N, 4)
            weights (np.ndarray): Weights for each quaternion (N,)
            
        Returns:
            np.ndarray: Mean quaternion (4,)
        """
        # Normalize all quaternions
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Ensure consistent quaternion representation (positive scalar part)
        for i in range(quaternions.shape[0]):
            if quaternions[i, 0] < 0:
                quaternions[i] = -quaternions[i]
        
        # Weighted average
        mean_quat = np.average(quaternions, axis=0, weights=weights)
        
        # Normalize result
        mean_quat = mean_quat / np.linalg.norm(mean_quat)
        
        return mean_quat

    def _quaternion_deviation(self, q1, q2):
        """
        Compute quaternion deviation (q1 - q2) in the tangent space.
        
        Parameters:
            q1 (np.ndarray): First quaternion (4,)
            q2 (np.ndarray): Second quaternion (4,)
            
        Returns:
            np.ndarray: Deviation vector (3,) in tangent space
        """
        # Ensure consistent representation
        if q1[0] < 0:
            q1 = -q1
        if q2[0] < 0:
            q2 = -q2
            
        # Compute relative quaternion
        q_rel = self.quaternion_multiply(q1, self.quaternion_conjugate(q2))
        
        # Convert to axis-angle representation
        angle = 2 * np.arccos(np.clip(abs(q_rel[0]), 0, 1))
        if angle < 1e-8:
            return np.zeros(3)
        
        axis = q_rel[1:4] / np.sin(angle/2)
        return axis * angle

    @staticmethod
    def quaternion_conjugate(q):
        """Compute quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def predict_measurement(self, sigma_points_pred):
        """Predict [altitude, vertical velocity] from predicted sigma points.
        Returns:
            z_pred (np.ndarray): (2,) predicted measurement mean [z, vz]
            S (np.ndarray): (2,2) measurement covariance
            Z_sigma (np.ndarray): (2n+1,2) sigma-point measurements
        """
        n_sigma = sigma_points_pred.shape[0]
        Z_sigma = np.zeros((n_sigma, 2))

        for i, sigma in enumerate(sigma_points_pred):
            z_pred_i  = sigma[2]  # altitude
            vz_pred_i = sigma[5]  # vertical velocity
            Z_sigma[i, 0] = z_pred_i
            Z_sigma[i, 1] = vz_pred_i

        # Predicted mean measurement
        z_pred = (self.Wm @ Z_sigma).reshape(2,)

        # Measurement covariance
        S = np.zeros((2, 2))
        for i in range(n_sigma):
            dz = (Z_sigma[i] - z_pred).reshape(2,1)
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

        # Previous published accel for delta calculation
        prev = self.filtered_accel_body if np.any(self.filtered_accel_body) else acc_med
        delta = acc_med - prev

        # --- PARACHUTE-SPECIFIC CLAMPING ---
        if self.apogee_detected:
            # Much tighter clamping during parachute descent
            parachute_clamp = 1.5  # m/s² max change per step during parachute
            delta = np.clip(delta, -parachute_clamp, parachute_clamp)
        else:
            # Normal operation clamping
            acc_mag_tmp = float(np.linalg.norm(acc_med))
            dev_from_1g_tmp = abs(acc_mag_tmp - 9.81)
            clamp_xy = self.acc_soft_jump_max
            clamp_z  = self.acc_soft_jump_max if dev_from_1g_tmp <= self.acc_thrust_gate_dev else self.acc_soft_jump_max_z_thrust
            delta[0] = np.clip(delta[0], -clamp_xy, clamp_xy)
            delta[1] = np.clip(delta[1], -clamp_xy, clamp_xy)
            delta[2] = np.clip(delta[2], -clamp_z, clamp_z)

        accel_prefiltered = prev + delta

        # --- Gravity model in BODY frame ---
        g_world = np.array([0, 0, -9.81])
        q = self.x[6:10]
        g_body = self.rotate_vector(self.quaternion_conjugate(q), g_world)
        gravity_only = -g_body
        
        # --- IMPROVED DYNAMIC BLEND FOR PARACHUTE ---
        if self.apogee_detected:
            # During parachute: trust gravity model much more (90%)
            model_w = 0.9
            meas_w = 0.1
        else:
            # Normal dynamic blend
            accel_mag = np.linalg.norm(accel_prefiltered)
            dev_from_1g = abs(accel_mag - 9.81)
            k = 7.0
            meas_w = np.clip(dev_from_1g / k, 0.0, 1.0)
            model_w = np.minimum(1.0 - meas_w, self.max_model_blend)

        blended_accel = model_w * gravity_only + meas_w * accel_prefiltered

        # --- ADAPTIVE EMA SMOOTHING ---
        if self.apogee_detected:
            # Much stronger smoothing during parachute
            accel_alpha = 0.4  # Higher = more smoothing
            gyro_alpha = 0.5
        else:
            accel_alpha = self.accel_ema_alpha
            gyro_alpha = self.gyro_ema_alpha

        # EMA smoothing
        self._accel_ema = accel_alpha * self._accel_ema + (1.0 - accel_alpha) * blended_accel
        self._gyro_ema = gyro_alpha * self._gyro_ema + (1.0 - gyro_alpha) * gyro_meas_body
        
        self.filtered_accel_body = self._accel_ema.copy()
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

    def update_flight_phase(self, current_velocity, current_altitude, current_time):
        """
        Detect flight phase and adjust filter parameters accordingly
        """
        # Update apogee detection first
        self.detect_apogee(current_velocity, current_altitude, current_time)
        
        if self.apogee_detected:
            self.flight_phase = "descent"
            self.Q = self.Q_descent  # Use descent process noise
            
            # Also increase measurement noise during parachute oscillations
            self.R = np.diag([[5.0**2, 6.0**2]])  # Higher noise during descent
        else:
            if abs(current_velocity) > 10.0:  # High velocity = powered ascent
                self.flight_phase = "ascent"
                self.Q = self.Q_ascent
                self.R = np.diag([[2.0**2, 2.5**2]])
            else:  # Low velocity = coasting
                self.flight_phase = "coast"
                # Use intermediate noise values
                self.Q = self.Q_ascent * 2.0
                self.R = np.diag([[3.0**2, 3.5**2]])

    def update_adaptive_noise(self):
        """
        Adjust process and measurement noise based on flight phase
        """
        if self.thrust_phase == "high_thrust":
            # Higher process noise during high thrust (rapid changes)
            self.Q = np.diag([
                0.005, 0.005, 0.005,   # position
                0.05, 0.05, 0.05,      # velocity  
                0.0005, 0.0005, 0.0005, 0.0005,  # quaternion
                0.005, 0.005, 0.005    # angular velocity
            ])
            self.R = np.diag([[2.0**2, 3.0**2]])  # Lower measurement noise
            
        elif self.thrust_phase == "parachute":
            # Higher measurement noise during parachute (oscillations)
            self.Q = np.diag([
                0.002, 0.002, 0.002,
                0.02, 0.02, 0.02, 
                0.0002, 0.0002, 0.0002, 0.0002,
                0.002, 0.002, 0.002
            ])
            self.R = np.diag([[5.0**2, 6.0**2]])  # Higher measurement noise
            
        else:  # coasting, low_thrust, mid_thrust
            # Default noise parameters
            self.Q = np.diag([
                0.001, 0.001, 0.001,
                0.01, 0.01, 0.01,
                0.0001, 0.0001, 0.0001, 0.0001,
                0.001, 0.001, 0.001
            ])
            self.R = np.diag([[2.0**2, 2.5**2]])
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

    def get_world_acceleration(self, prev_vel=None, dt=None, current_time=0.0):
        """
        Phase-aware world acceleration estimate:
            - Derivative path: central difference of EMA-smoothed velocity
            - Attitude path : rotate filtered body accel + add gravity
            - Crossfade weights: thrust detector + apogee-time decay
        Parameters:
            prev_vel (np.ndarray): Previous velocity (3,), optional
            dt (float): Time step, optional
            current_time (float): Current time in seconds
            
        Returns:
            np.ndarray: World acceleration (3,) in m/s²
        """
        current_vel = self.x[3:6].copy()
        quat = self.x[6:10].copy()
        
        # Method 1: Attitude-based 
        f_world = self.rotate_vector(quat, self.filtered_accel_body)
        a_att = f_world + np.array([0.0, 0.0, -9.81])
        
        # Method 2: Raw velocity derivative (NO EMA for thrust phases)
        a_vel = None
        if prev_vel is not None and dt is not None and dt > 0:
            # Use RAW velocity difference (no filtering) for maximum responsiveness
            vel_diff = current_vel - prev_vel
            a_vel_raw = vel_diff / dt
            
            # Very permissive validation during thrust
            max_phys_accel = 200.0  # Allow very high acceleration during thrust
            if np.all(np.abs(a_vel_raw) < max_phys_accel):
                a_vel = a_vel_raw
            else:
                a_vel = a_att
        
        # --- ULTRA-AGGRESSIVE BLENDING FOR THRUST ---
        accel_mag = np.linalg.norm(self.filtered_accel_body)
        dev_from_1g = abs(accel_mag - 9.81)
        
        if self.apogee_detected:
            blend_weight = 0.05  # 95% attitude during parachute
            
        elif dev_from_1g > 20.0:  # Extreme thrust (liftoff)
            blend_weight = 0.99   # 99% derivative
            
        elif dev_from_1g > 10.0:  # High thrust  
            blend_weight = 0.97   # 97% derivative
            
        elif dev_from_1g > 5.0:   # Medium thrust
            blend_weight = 0.92   # 92% derivative
            
        elif dev_from_1g > 2.0:   # Low thrust
            blend_weight = 0.85   # 85% derivative
            
        else:                     # Coasting
            blend_weight = 0.3    # 30% derivative
        
        # Final blend
        if a_vel is not None:
            a_world = blend_weight * a_vel + (1.0 - blend_weight) * a_att
        else:
            a_world = a_att
        
        return a_world

    def detect_apogee(self, current_velocity, current_altitude, current_time):
        """
        Detect apogee (peak altitude) for parachute deployment handling.
        
        Parameters:
            current_velocity (float): Current vertical velocity in m/s
            current_altitude (float): Current altitude in m
            current_time (float): Current time in seconds
        """
        # Track velocity and altitude history
        self.velocity_history.append(current_velocity)
        self.altitude_history.append(current_altitude)
        if len(self.velocity_history) > self.velocity_window:
            self.velocity_history.pop(0)
            self.altitude_history.pop(0)
        
        # Detect apogee: velocity crosses from positive to negative AND altitude is near peak
        if (not self.apogee_detected and 
            len(self.velocity_history) >= 3 and 
            current_velocity < 0 and 
            self.velocity_history[-2] > 0 and
            current_time > 5.0):  # Must be after thrust phase
            
            # Additional check: make sure altitude is actually at a peak
            if len(self.altitude_history) >= 3:
                recent_altitudes = self.altitude_history[-3:]
                if (recent_altitudes[-1] < recent_altitudes[-2] and 
                    recent_altitudes[-2] > recent_altitudes[-3]):
                    
                    self.apogee_detected = True
                    self.apogee_time = current_time
                    print(f"Apogee detected at t={current_time:.2f}s, v={current_velocity:.2f} m/s, h={current_altitude:.1f} m")

    def detect_thrust_phase(self, accel_measured, current_time):
        """
        Detect different thrust phases for better filter tuning
        """
        accel_mag = np.linalg.norm(accel_measured)
        dev_from_1g = abs(accel_mag - 9.81)
        
        # Update thrust phase classification
        if self.apogee_detected:
            self.thrust_phase = "parachute"
        elif dev_from_1g > 15.0:
            self.thrust_phase = "high_thrust"
        elif dev_from_1g > 8.0:
            self.thrust_phase = "mid_thrust" 
        elif dev_from_1g > 3.0:
            self.thrust_phase = "low_thrust"
        else:
            self.thrust_phase = "coasting"

    def get_world_acceleration_from_velocity(self, prev_vel=None, dt=None):
        """
        Get world acceleration by differentiating velocity (alternative method).
        
        Parameters:
            prev_vel (np.ndarray): Previous velocity (3,), optional
            dt (float): Time step, optional
            
        Returns:
            np.ndarray: World acceleration (3,) in m/s²
        """
        if prev_vel is None or dt is None:
            # Fallback to the other method
            return self.get_world_acceleration()
        
        # Compute acceleration as velocity derivative
        vel = self.x[3:6]
        a_world = (vel - prev_vel) / dt
        
        return a_world

