import numpy as np

class UKF:
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

        # --- Measurement noise covariance (accel + gyro + pressure) ---
        self.R = np.eye(7)
        self.R[:6, :6] *= 0.5    # accel + gyro noise
        self.R[6, 6] = 5.0       # barometric pressure noise

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

            # --- Rotate measured accel to WORLD frame for integration ---
            # NOTE: rotate_vector(q, v) below implements q * [0,v] * q_conj.
            # Here we want BODY->WORLD, so we use the same function as defined.
            acc_world = self.rotate_vector(quat, accel_measured)

            # --- Kinematics integration (semi-implicit Euler) ---
            vel_new = vel + acc_world * dt
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
        """
        Predict sensor measurements from predicted sigma points.

        Parameters:
            sigma_points_pred (np.ndarray): Predicted sigma points (2n+1, n).

        Returns:
            z_pred (np.ndarray): Predicted measurement mean (7,).
            S (np.ndarray): Measurement covariance matrix (7, 7).
            Z_sigma (np.ndarray): Sigma point measurements (2n+1, 7).
        """
        n_sigma = sigma_points_pred.shape[0]
        Z_sigma = np.zeros((n_sigma, 7))  # accel(3), gyro(3), pressure(1)

        for i, sigma in enumerate(sigma_points_pred):
            vel  = sigma[3:6]
            quat = sigma[6:10]
            omega = sigma[10:13]

            # --- IMU accel model: gravity only in BODY (we rely on process for dynamics) ---
            g_world = np.array([0, 0, -9.81])
            g_body  = self.rotate_vector(quat, g_world)
            accel_pred = -g_body  # what an ideal stationary accelerometer would read

            # --- Pressure model from altitude (z position) ---
            pressure_pred = (self.P0 * np.exp(-sigma[2] / self.H) - self.P0) / 10.0

            # Fill measurement for this sigma point
            Z_sigma[i, 0:3] = accel_pred
            Z_sigma[i, 3:6] = omega
            Z_sigma[i, 6]   = pressure_pred

        # --- Predicted mean measurement ---
        z_pred = np.dot(self.Wm, Z_sigma)

        # --- Measurement covariance ---
        S = np.zeros((7, 7))
        for i in range(n_sigma):
            dz = Z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
        S += self.R  # add measurement noise

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
        # --- Gravity model in BODY frame from current attitude ---
        g_world = np.array([0, 0, -9.81])
        g_body  = self.rotate_vector(self.x[6:10], g_world)
        gravity_only = -g_body  # what a perfect stationary accelerometer would read

        # --- Dynamic blend weight based on how "non-1g" we are ---
        accel_mag = np.linalg.norm(accel_meas_body)
        dev_from_1g = abs(accel_mag - 9.81)                   # how far from 1g
        # Map deviation to [0..1]; 0 -> fully model, big deviation -> fully measurement
        k = 4.0  # [m/s^2] deviation for ~100% measurement; tune as needed
        meas_w = np.clip(dev_from_1g / k, 0.0, 1.0)
        # Cap model contribution so we never over-trust it
        model_w = np.minimum(1.0 - meas_w, self.max_model_blend)

        # --- Blended accel before EMA ---
        blended_accel = model_w * gravity_only + (1.0 - model_w) * accel_meas_body

        # --- EMA smoothing (accel) ---
        a = self.accel_ema_alpha
        self._accel_ema = a * self._accel_ema + (1.0 - a) * blended_accel
        self.filtered_accel_body = self._accel_ema.copy()

        # --- EMA smoothing (gyro) ---
        g = self.gyro_ema_alpha
        self._gyro_ema = g * self._gyro_ema + (1.0 - g) * gyro_meas_body
        self.filtered_gyro_body = self._gyro_ema.copy()

    def update(self, x_pred, P_pred, sigma_points_pred, z_pred, S, Z_sigma, z_measured):
        """
        Perform Kalman update step with actual measurement.

        Parameters:
            x_pred (np.ndarray): Predicted state mean.
            P_pred (np.ndarray): Predicted state covariance.
            sigma_points_pred (np.ndarray): Predicted sigma points.
            z_pred (np.ndarray): Predicted measurement mean (7,).
            S (np.ndarray): Measurement covariance matrix (7, 7).
            Z_sigma (np.ndarray): Sigma point measurements (2n+1, 7).
            z_measured (np.ndarray): Actual sensor measurement (7,) = [accel(3), gyro(3), pressure(1)].

        Returns:
            None. Updates self.x and self.P in place and refreshes filtered sensor outputs.
        """
        n_z = z_measured.shape[0]
        P_xz = np.zeros((self.n, n_z))

        # --- Cross covariance between state and measurement ---
        for i in range(sigma_points_pred.shape[0]):
            dx = sigma_points_pred[i] - x_pred
            dz = Z_sigma[i] - z_pred
            P_xz += self.Wc[i] * np.outer(dx, dz)

        # --- Kalman gain ---
        K = P_xz @ np.linalg.inv(S)

        # --- Innovation ---
        dz = z_measured - z_pred

        # --- State & covariance update ---
        self.x = x_pred + K @ dz
        self.P = P_pred - K @ S @ K.T

        # --- After state update, create UKF-aided denoised sensor outputs for plotting ---
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
