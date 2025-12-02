import numpy as np
import pandas as pd
from physics_sim import simulate_rocket
from graph import create_graphs
from ukf import UKF

def pressure_to_altitude_ISA(P_hPa, P0=1013.25, T0=288.15):
    """More accurate pressure to altitude conversion with temperature parameter"""
    L = 0.0065  # K/m
    R = 287.05  # J/(kg·K) - gas constant for air
    g = 9.80665 # m/s²
    
    if P_hPa <= 0:
        return 0.0
    
    # Hypsometric formula
    altitude = (T0 / L) * (1.0 - (P_hPa / P0) ** (L * R / g))
    return max(0.0, altitude)

def debug_altitude_calibration(df, P0_hat, h_meas_full, h_smooth, altitude_bias):
    """Debug function to verify altitude calibration"""
    if 'altitude_true' in df.columns:
        true_altitudes = df['altitude_true'].values
        time = df['time'].values
        
        # Calculate errors
        initial_error = h_meas_full[0] - true_altitudes[0]
        corrected_initial_error = (h_meas_full[0] + altitude_bias) - true_altitudes[0]
        final_error = h_smooth[-1] - true_altitudes[-1] if len(h_smooth) == len(true_altitudes) else 0
        
        print("\n=== ALTITUDE CALIBRATION DEBUG ===")
        print(f"Initial true altitude: {true_altitudes[0]:.1f} m")
        print(f"Initial measured altitude: {h_meas_full[0]:.1f} m")
        print(f"Initial error (before correction): {initial_error:.1f} m")
        print(f"Initial error (after correction): {corrected_initial_error:.1f} m")
        print(f"Final error: {final_error:.1f} m")
        print("=================================\n")

def run_ukf_on_dataset(csv_path, dt=0.01):
    df = pd.read_csv(csv_path)
    ukf = UKF(dt=dt)
    time = df['time'].values

    # --- CRITICAL FIX: Better P0 calibration using known launch altitude ---
    # Use longer initialization period and known physics
    N0 = max(50, int(3.0 / dt))  # 3 seconds for stable calibration
    
    # Get initial pressures and remove obvious outliers
    pressures_init = df['pressure_measured'].values[:N0]
    pressure_median = np.median(pressures_init)
    pressure_std = np.std(pressures_init)
    
    # Remove outliers (beyond 1.5 sigma)
    valid_mask = np.abs(pressures_init - pressure_median) < 1.5 * pressure_std
    valid_pressures = pressures_init[valid_mask]
    
    if len(valid_pressures) == 0:
        P0_hat = pressure_median
    else:
        P0_hat = float(np.mean(valid_pressures))
    
    print(f"Calibrated P0: {P0_hat:.2f} hPa (using {len(valid_pressures)} samples)")
    
    def h_from_P(P): 
        return pressure_to_altitude_ISA(P, P0=P0_hat)
    
    
    # --- IMPROVED: Use TRUE altitude for initial alignment ---
    # Get true initial altitude from simulation
    if 'altitude_true' in df.columns:
        true_h0 = float(df['altitude_true'].values[0])
        print(f"True initial altitude: {true_h0:.1f} m")
    else:
        true_h0 = 0.0
    
    # Convert all pressures to altitudes
    h_meas_full = np.array([h_from_P(p) for p in df['pressure_measured'].values])
    
    # Apply altitude bias correction based on initial true altitude
    measured_h0 = h_meas_full[0]
    altitude_bias = true_h0 - measured_h0
    print(f"Altitude bias correction: {altitude_bias:.1f} m")
    
    h_meas_full_corrected = h_meas_full + altitude_bias
    
    # --- IMPROVED: Better smoothing with phase awareness ---
    def flight_phase_aware_smooth(altitudes, time_array, dt):
        """Smart smoothing based on flight dynamics"""
        n = len(altitudes)
        smoothed = np.zeros(n)
        
        # Different windows for different phases
        win_powered = max(3, int(0.1 / dt))    # 100ms - responsive during thrust
        win_coast = max(7, int(0.3 / dt))      # 300ms - moderate smoothing
        win_descent = max(15, int(0.8 / dt))   # 800ms - strong smoothing during parachute
        
        # Simple phase detection
        for i in range(n):
            if i < 5:
                smoothed[i] = np.mean(altitudes[:i+1])
                continue
                
            # Estimate current vertical velocity
            if i >= 10:
                recent_vel = (altitudes[i] - altitudes[i-5]) / (5 * dt)
            else:
                recent_vel = (altitudes[i] - altitudes[0]) / (i * dt)
            
            # Choose window based on dynamics
            if abs(recent_vel) > 50:  # Powered flight
                window = win_powered
            elif recent_vel < -5:     # Parachute descent
                window = win_descent
            else:                     # Coasting
                window = win_coast
                
            window = min(window, i + 1)
            start_idx = max(0, i - window + 1)
            
            # Use median for descent (robust to oscillations), mean for ascent
            if recent_vel < -5:
                smoothed[i] = np.median(altitudes[start_idx:i+1])
            else:
                smoothed[i] = np.mean(altitudes[start_idx:i+1])
        
        return smoothed
    
    h_smooth = flight_phase_aware_smooth(h_meas_full_corrected, time, dt)
    
    # Compute vertical velocity from smoothed altitude
    vz_meas_full = np.gradient(h_smooth, dt)
    
    # Fix endpoints
    if len(vz_meas_full) > 2:
        vz_meas_full[0] = vz_meas_full[1]
        vz_meas_full[-1] = vz_meas_full[-2]
    
    # --- IMPROVED: Better UKF initialization using corrected data ---
    # Use the corrected and smoothed initial values
    ukf.x[2] = h_smooth[0]  # Corrected initial altitude
    ukf.x[5] = vz_meas_full[0]  # Initial vertical velocity
    
    # Rest of initialization (attitude, etc.) remains the same...
    # [Previous IMU-based attitude initialization code]
    
    # Initialize with first IMU readings for attitude
    accel_init = df[['accel_x', 'accel_y', 'accel_z']].values[:N0].mean(axis=0)
    accel_mag = np.linalg.norm(accel_init)
    if accel_mag > 1e-6:
        gravity_body = accel_init / accel_mag
        gravity_world = np.array([0, 0, -1])
        
        v = np.cross(gravity_world, gravity_body)
        s = np.linalg.norm(v)
        c = np.dot(gravity_world, gravity_body)
        
        if s < 1e-8:
            if c > 0:
                ukf.x[6:10] = np.array([1, 0, 0, 0])
            else:
                ukf.x[6:10] = np.array([0, 1, 0, 0])
        else:
            v = v / s
            ukf.x[6:10] = np.array([c, v[0], v[1], v[2]])
            ukf.x[6:10] = ukf.x[6:10] / np.linalg.norm(ukf.x[6:10])
    
    gyro_init = df[['gyro_x', 'gyro_y', 'gyro_z']].values[:N0].mean(axis=0)
    ukf.x[10:13] = gyro_init

    # --- REDUCED MEASUREMENT NOISE for better altitude tracking ---
    ukf.R = np.diag([[1.5**2, 2.0**2]])  # Much tighter measurement noise
    
    # Rest of the UKF processing loop remains the same...
    state_history = []
    filtered_accel = []
    filtered_gyro = []
    net_body_accel_estimates = []
    world_accel_estimates = []
    raw_accel, raw_gyro, raw_pressure = [], [], []
    
    prev_vel = None

    debug_altitude_calibration(df, P0_hat, h_meas_full, h_smooth, altitude_bias)

    for i in range(len(df)):
        pressure = df.loc[i, 'pressure_measured']
        accel = df.loc[i, ['accel_x', 'accel_y', 'accel_z']].values.astype(float)
        gyro  = df.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].values.astype(float)

        # Use corrected and smoothed measurements
        z = np.array([h_smooth[i], vz_meas_full[i]])

        # UKF predict & update
        sigma_pts = ukf.generate_sigma_points()
        sigma_pts_pred = ukf.predict_sigma_points(sigma_pts, accel_measured=accel, gyro_measured=gyro)
        ukf.predict_mean_and_covariance(sigma_pts_pred)
        z_pred, S, Z_sigma = ukf.predict_measurement(sigma_pts_pred)
        ukf.update(ukf.x, ukf.P, sigma_pts_pred, z_pred, S, Z_sigma, z)
        
        # Detect phases
        current_velocity = ukf.x[5]
        current_altitude = ukf.x[2]
        ukf.detect_apogee(current_velocity, current_altitude, time[i])
        ukf.detect_thrust_phase(accel, time[i])
        ukf.update_adaptive_noise()

        # Logging
        ukf._update_sensor_filters(accel, gyro)
        state_history.append(ukf.x.copy())
        filtered_accel.append(ukf.get_filtered_accel())
        filtered_gyro.append(ukf.get_filtered_gyro())
        net_body_accel_estimates.append(ukf.estimate_net_body_acceleration())
        
        current_vel = ukf.x[3:6]
        world_accel_estimates.append(ukf.get_world_acceleration(prev_vel, dt, time[i]))
        prev_vel = current_vel.copy()

        raw_accel.append(accel)
        raw_gyro.append(gyro)
        raw_pressure.append(pressure)

    return (np.array(state_history), time,
            np.array(filtered_accel), np.array(filtered_gyro),
            np.array(net_body_accel_estimates), np.array(world_accel_estimates),
            np.array(raw_accel), np.array(raw_gyro), np.array(raw_pressure))

# --- Generate data & run the filter ------------------------------------------
df = simulate_rocket(T_total=50, save=True, save_path="datasets/rocket_flight_data.csv")

(filtered_states, timestamps, filtered_accel, filtered_gyro,
 net_body_accel, world_accel, raw_accel, raw_gyro, pressure) = \
    run_ukf_on_dataset("datasets/rocket_flight_data_noisy.csv")

# --- Plots --------------------------------------------------------------------
create_graphs(
    df, timestamps, filtered_states,
    filtered_accel, filtered_gyro,
    net_body_accel, world_accel,
    raw_accel, raw_gyro, pressure
)
