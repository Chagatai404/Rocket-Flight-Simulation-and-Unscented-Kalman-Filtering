import numpy as np
import pandas as pd
import os

def simulate_rocket(T_total=50, dt=0.01, save=False, save_path="datasets/rocket_flight_data.csv"):
    # === Constants ===
    g = 9.81             # gravity (m/s²)
    rho = 1.225          # air density (kg/m³)
    Cd = 0.4             # drag coefficient (streamlined)
    A = 0.03             # cross-sectional area (m²)
    m0 = 3.0             # initial mass (kg)
    mass_loss = 1.5      # kg of fuel to burn
    burn_time = 11.0     # burn duration in seconds
    thrust = 360.0       # thrust in Newtons
    ramp_time = 0.5      # ramp-down period for thrust

    # === Time and state arrays ===
    n = int(T_total / dt)
    time = np.linspace(0, T_total, n)
    h = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    mass = np.full(n, m0)

    # Simulate linear mass loss during thrust
    for i in range(1, int(burn_time / dt)):
        mass[i] = m0 - mass_loss * (i * dt / burn_time)

    # === Simulation loop ===
    for i in range(1, n):
        v_prev = v[i - 1]
        m = mass[i]

        # Drag force
        F_drag = 0.5 * rho * Cd * A * v_prev**2 * np.sign(v_prev)

        # Thrust with ramp-down smoothing
        if time[i] <= burn_time:
            F_thrust = thrust
        elif burn_time < time[i] <= burn_time + ramp_time:
            F_thrust = thrust * (1 - (time[i] - burn_time)/ramp_time)
        else:
            F_thrust = 0

        # Net force and acceleration
        net_force = F_thrust - F_drag - m * g
        a[i] = net_force / m
        v[i] = v_prev + a[i] * dt
        h[i] = h[i - 1] + v[i] * dt

    # === Ground truth DataFrame ===
    data = pd.DataFrame({
        'time': time,
        'altitude_true': h,
        'velocity_true': v,
        'acceleration_true': a,
        'mass': mass
    })

    # === Sensor noise models ===
    accel_noise_std = 1.0      # m/s²
    gyro_noise_std = 0.05      # rad/s
    pressure_noise_std = 5.0   # hPa

    # Accelerometer (z-axis, includes gravity)
    accel_measured = a + np.random.normal(0, accel_noise_std, size=n)

    # Gyroscope (dummy: assume no rotation, only noise)
    gyro_measured = np.random.normal(0, gyro_noise_std, size=n)

    # Drift and bias simulation
    def generate_bias_drift(n, bias_mean, bias_std, drift_rate):
        base_bias = np.random.normal(bias_mean, bias_std)
        drift = np.linspace(0, drift_rate, n)  # linear drift
        return base_bias + drift

    # Apply to accelerometer and gyro
    accel_drift = generate_bias_drift(n, bias_mean=0.5, bias_std=0.2, drift_rate=0.5)
    gyro_drift = generate_bias_drift(n, bias_mean=0.01, bias_std=0.01, drift_rate=0.02)

    # Add drift to noisy signals
    accel_measured += accel_drift
    gyro_measured += gyro_drift

    # Add outliers to simulate vibrations or electromagnetic interference
    def inject_outliers(signal, num_outliers=10, magnitude=5.0):
        idx = np.random.choice(len(signal), size=num_outliers, replace=False)
        signal[idx] += np.random.normal(0, magnitude, size=num_outliers)
        return signal

    accel_measured = inject_outliers(accel_measured, num_outliers=20, magnitude=10)
    gyro_measured = inject_outliers(gyro_measured, num_outliers=20, magnitude=0.1)

    # Barometric pressure from altitude using ISA model
    P0 = 1013.25  # hPa
    T0 = 288.15   # K
    L = 0.0065    # K/m
    R = 8.31447
    M = 0.0289644
    g0 = 9.80665

    pressure_true = P0 * (1 - (L * h) / T0) ** ((g0 * M) / (R * L))
    pressure_measured = pressure_true + np.random.normal(0, pressure_noise_std, size=n)

    # === Add noisy sensor readings ===
    data['accel_measured'] = accel_measured
    data['gyro_measured'] = gyro_measured
    data['pressure_measured'] = pressure_measured

    # === Save files ===
    if save:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save clean data (ground truth only)
        clean_data = data[['time', 'altitude_true', 'velocity_true', 'acceleration_true', 'mass']]
        clean_path = save_path.replace(".csv", "_clean.csv")
        clean_data.to_csv(clean_path, index=False)
        print(f"✅ Saved clean dataset to '{clean_path}'")

        # Save noisy dataset (all columns)
        noisy_path = save_path.replace(".csv", "_noisy.csv")
        data.to_csv(noisy_path, index=False)
        print(f"✅ Saved noisy dataset to '{noisy_path}'")

    return data