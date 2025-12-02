# Simulate a 1D rocket flight with full 3D IMU (accelerometer + gyro) and quaternion orientation
import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R

def simulate_rocket(T_total=50, dt=0.01, save=False, save_path="datasets/rocket_flight_data.csv"):
    # === Constants for physics ===
    g = 9.81  # gravity (m/s^2)
    rho = 1.225  # air density (kg/m^3)
    Cd = 0.4  # drag coefficient
    A = 0.03  # cross-sectional area (m^2)
    m0 = 3.0  # initial mass (kg)
    mass_loss = 1.5  # total fuel mass (kg)
    burn_time = 11.0  # burn duration (s)
    thrust = 360.0  # thrust (N)
    ramp_time = 0.5  # thrust ramp down period (s)

    # === Time array ===
    n = int(T_total / dt)
    time = np.linspace(0, T_total, n)

    # === Initialize rocket state arrays ===
    h = np.zeros(n)  # altitude
    v = np.zeros(n)  # velocity
    a = np.zeros(n)  # acceleration
    mass = np.full(n, m0)

    # === Simulate fuel burn: linear mass loss ===
    for i in range(1, int(burn_time / dt)):
        mass[i] = m0 - mass_loss * (i * dt / burn_time)

    # === Physics simulation loop ===
    for i in range(1, n):
        v_prev = v[i - 1]
        m = mass[i]

        # Drag force
        F_drag = 0.5 * rho * Cd * A * v_prev**2 * np.sign(v_prev)

        # Thrust logic with ramp-down
        if time[i] <= burn_time:
            F_thrust = thrust
        elif burn_time < time[i] <= burn_time + ramp_time:
            F_thrust = thrust * (1 - (time[i] - burn_time)/ramp_time)
        else:
            F_thrust = 0

        # Net force and motion integration
        net_force = F_thrust - F_drag - m * g
        a[i] = net_force / m
        v[i] = v_prev + a[i] * dt
        h[i] = h[i - 1] + v[i] * dt

    # === Simulate quaternion orientation over time ===
    # Create artificial angular velocities around roll/pitch/yaw axes
    angular_rates = np.deg2rad(np.vstack([
        2 * np.sin(0.2 * time),     # roll rate (x-axis)
        1.5 * np.cos(0.1 * time),   # pitch rate (y-axis)
        1.0 * np.sin(0.3 * time)    # yaw rate (z-axis)
    ])).T

    # Integrate orientation using scipy's Rotation
    quaternions = []
    r = R.identity()
    for i in range(n):
        rotvec = angular_rates[i] * dt
        r = r * R.from_rotvec(rotvec)
        quaternions.append(r.as_quat())  # [x, y, z, w]

    quaternions = np.array(quaternions)
    q_x, q_y, q_z, q_w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # === Simulate 3D accelerometer readings ===
# IMPORTANT: The simulated IMU outputs *specific force* in BODY coordinates,
# i.e., measured_accel_body = R_world_to_body( linear_accel_world − gravity_world ).
# This matches real accelerometers and the UKF prediction step which rotates
# BODY→WORLD and ADDS gravity back before integrating.
    # Acceleration in world frame (z only), plus gravity
    acc_world = np.zeros((n, 3))
    acc_world[:, 2] = a
    gravity_world = np.array([0, 0, -g])

    # Rotate acceleration + gravity into body frame
    acc_body = np.zeros_like(acc_world)
    for i in range(n):
        r = R.from_quat(quaternions[i])
        acc_body[i] = r.apply(acc_world[i] + gravity_world)

    # === Simulate gyroscope readings from quaternion differences ===
    gyro_body = np.zeros((n, 3))
    for i in range(1, n):
        r_prev = R.from_quat(quaternions[i - 1])
        r_curr = R.from_quat(quaternions[i])
        delta_r = r_curr * r_prev.inv()
        gyro_body[i] = delta_r.as_rotvec() / dt

    # === Add sensor noise, bias drift, and outliers ===

    # Bias + drift generator
    def generate_bias_drift(n, bias_mean, bias_std, drift_rate):
        base_bias = np.random.normal(bias_mean, bias_std, size=(n, 1))
        drift = np.linspace(0, drift_rate, n).reshape(-1, 1)
        return base_bias + drift

    # Outlier injector to simulate vibrations or EMI
    def inject_outliers(signal, num_outliers=20, magnitude=10.0):
        for dim in range(signal.shape[1]):
            idx = np.random.choice(signal.shape[0], size=num_outliers, replace=False)
            signal[idx, dim] += np.random.normal(0, magnitude, size=num_outliers)
        return signal

    # Noise levels
    accel_noise_std = 1.0
    gyro_noise_std = 0.05

    # Add bias + drift + noise
    accel_drift = generate_bias_drift(n, 0.5, 0.2, 0.5)
    gyro_drift = generate_bias_drift(n, 0.01, 0.01, 0.02)

    accel_measured = acc_body + np.random.normal(0, accel_noise_std, acc_body.shape) + accel_drift
    gyro_measured = gyro_body + np.random.normal(0, gyro_noise_std, gyro_body.shape) + gyro_drift

    # Inject occasional large spikes 
    accel_measured = inject_outliers(accel_measured, num_outliers=20, magnitude=10)
    gyro_measured = inject_outliers(gyro_measured, num_outliers=20, magnitude=0.1)

    # === Simulate barometric pressure sensor ===
    P0, T0, L = 1013.25, 288.15, 0.0065
    R_gas, M, g0 = 8.31447, 0.0289644, 9.80665
    pressure_true = P0 * (1 - (L * h) / T0) ** ((g0 * M) / (R_gas * L))
    pressure_measured = pressure_true + np.random.normal(0, 5.0, size=n)

    # === Build final DataFrame ===
    # Also expose *true* IMU signals (before noise/drift/outliers) so we can compare UKF performance
    # against ground truth:
    accel_true = acc_body.copy()
    gyro_true  = gyro_body.copy()

    data = pd.DataFrame({
        'time': time,
        'altitude_true': h,
        'velocity_true': v,
        'acceleration_true': a,
        'pressure_true': pressure_true,
        'accel_true_x': accel_true[:,0], 'accel_true_y': accel_true[:,1], 'accel_true_z': accel_true[:,2],
        'gyro_true_x': gyro_true[:,0], 'gyro_true_y': gyro_true[:,1], 'gyro_true_z': gyro_true[:,2],
        'mass': mass,
        'accel_x': accel_measured[:, 0],
        'accel_y': accel_measured[:, 1],
        'accel_z': accel_measured[:, 2],
        'gyro_x': gyro_measured[:, 0],
        'gyro_y': gyro_measured[:, 1],
        'gyro_z': gyro_measured[:, 2],
        'pressure_measured': pressure_measured,
        'q_x': q_x,
        'q_y': q_y,
        'q_z': q_z,
        'q_w': q_w
    })

    # === Save clean and noisy datasets ===
    if save:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Clean dataset: ground truth only
        clean_cols = ['time','altitude_true','velocity_true','acceleration_true','mass','pressure_true','accel_true_x','accel_true_y','accel_true_z','gyro_true_x','gyro_true_y','gyro_true_z','q_x','q_y','q_z','q_w']
        clean_data = data[clean_cols]
        clean_path = save_path.replace(".csv", "_clean.csv")
        clean_data.to_csv(clean_path, index=False)

        # Noisy dataset: full sensor data
        noisy_path = save_path.replace(".csv", "_noisy.csv")
        data.to_csv(noisy_path, index=False)

    return data
