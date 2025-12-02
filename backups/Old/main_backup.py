import numpy as np
import pandas as pd
from physics_sim import simulate_rocket
from graph import create_graphs
from ukf import UKF

def run_ukf_on_dataset(csv_path, dt=0.01):
    """
    Run the UKF over the noisy dataset and collect time series for plotting.

    Returns:
        filtered_states (np.ndarray): UKF state history (N, 13).
        time (np.ndarray): timestamp array (N,).
        filtered_accel (np.ndarray): accel derived from UKF state (gravity-only) (N,3).
        filtered_gyro (np.ndarray): UKF-estimated angular rates (N,3).
        net_body_accel_estimates (np.ndarray): body-frame accel used in prediction (N,3).
        model_accel (np.ndarray): gravity-only model accel (same as what UKF expects) (N,3).
        raw_accel (np.ndarray): raw accelerometer samples (N,3).
        raw_gyro (np.ndarray): raw gyro samples (N,3).
        raw_pressure (np.ndarray): raw pressure samples (N,).
    """
    df = pd.read_csv(csv_path)
    ukf = UKF(dt=dt)

    state_history = []
    filtered_accel = []            # gravity-only accel derived from state orientation
    filtered_gyro = []             # UKF-estimated body rates
    net_body_accel_estimates = []  # accel used in prediction step (from sensor)

    raw_accel = []
    raw_gyro = []
    raw_pressure = []

    time = df['time'].values

    for i in range(len(df)):
        # --- Extract sensors ---------------------------------------------------
        pressure = df.loc[i, 'pressure_measured']
        # Normalize to match the measurement model used inside the UKF
        normalized_pressure = (pressure - 1013.25) / 10.0

        accel = df.loc[i, ['accel_x', 'accel_y', 'accel_z']].values.astype(float)  # body-frame accel
        gyro  = df.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].values.astype(float)     # body-frame rates

        # Measurement vector: [accel(3), gyro(3), pressure(1)]
        z = np.hstack([accel, gyro, [normalized_pressure]])

        # --- UKF predict & update ---------------------------------------------
        sigma_pts = ukf.generate_sigma_points()
        sigma_pts_pred = ukf.predict_sigma_points(sigma_pts, accel_measured=accel)
        ukf.predict_mean_and_covariance(sigma_pts_pred)
        z_pred, S, Z_sigma = ukf.predict_measurement(sigma_pts_pred)
        ukf.update(ukf.x, ukf.P, sigma_pts_pred, z_pred, S, Z_sigma, z)

        # --- Logging ---
        state_history.append(ukf.x.copy())
        filtered_accel.append(ukf.get_filtered_accel())
        filtered_gyro.append(ukf.get_filtered_gyro())
        net_body_accel_estimates.append(ukf.estimate_net_body_acceleration())

        raw_accel.append(accel)
        raw_gyro.append(gyro)
        raw_pressure.append(pressure)

    return (np.array(state_history), time,
            np.array(filtered_accel), np.array(filtered_gyro),
            np.array(net_body_accel_estimates),
            np.array(raw_accel), np.array(raw_gyro), np.array(raw_pressure))

# --- Generate data & run the filter ------------------------------------------
df = simulate_rocket(T_total=20, save=True, save_path="datasets/rocket_flight_data.csv")

(filtered_states, timestamps, filtered_accel, filtered_gyro,
 net_body_accel, raw_accel, raw_gyro, pressure) = \
    run_ukf_on_dataset("datasets/rocket_flight_data_noisy.csv")

# --- Plots --------------------------------------------------------------------
create_graphs(
    df, timestamps, filtered_states,
    filtered_accel, filtered_gyro,
    net_body_accel,
    raw_accel, raw_gyro, pressure
)
