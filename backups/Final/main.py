import numpy as np
import pandas as pd
from physics_sim import simulate_rocket
from graph import create_graphs
from ukf import UKF

def pressure_to_altitude_ISA(P_hPa, P0=1013.25):
    """Convert pressure (hPa) to altitude (m) using ISA standard atmosphere."""
    T0, L = 288.15, 0.0065
    R_gas, M, g0 = 8.31447, 0.0289644, 9.80665
    exponent = (R_gas * L) / (g0 * M)
    return (T0 / L) * (1.0 - (P_hPa / P0) ** exponent)

def run_ukf_on_dataset(csv_path, dt=0.01):
    """Run the UKF on a dataset and return arrays for plotting.

    Uses scalar altitude-only measurement (baro→altitude via ISA). Accel/gyro are
    treated as inputs to the process model. We self-calibrate P0 from the first
    ~0.5 s and also initialize altitude and vertical velocity for a tight start.
    """
    df = pd.read_csv(csv_path)
    ukf = UKF(dt=dt)

    time = df['time'].values

    # --- P0 bootstrap and initial z,vz ---------------------------------------
    N0 = max(5, int(0.5 / dt))
    P0_hat = float(np.median(df['pressure_measured'].values[:N0]))
    def h_from_P(P): return pressure_to_altitude_ISA(P, P0=P0_hat)
    h0 = float(np.mean([h_from_P(p) for p in df['pressure_measured'].values[:N0]]))
    h1 = float(np.mean([h_from_P(p) for p in df['pressure_measured'].values[N0:2*N0]]))
    v0 = (h1 - h0) / (N0 * dt)
    ukf.x[2] = h0
    ukf.x[5] = v0

    state_history = []
    filtered_accel = []            # gravity-only accel derived from state orientation
    filtered_gyro = []             # UKF-estimated body rates
    net_body_accel_estimates = []  # accel used in prediction (from sensor)
    raw_accel, raw_gyro, raw_pressure = [], [], []


    for i in range(len(df)):
        # --- Extract sensors ---------------------------------------------------
        pressure = df.loc[i, 'pressure_measured']

        accel = df.loc[i, ['accel_x', 'accel_y', 'accel_z']].values.astype(float)
        gyro  = df.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].values.astype(float)

        # Scalar altitude measurement via ISA inversion
        h_meas = h_from_P(pressure)
        z = np.array([h_meas])

        # --- UKF predict & update ---------------------------------------------
        sigma_pts = ukf.generate_sigma_points()
        sigma_pts_pred = ukf.predict_sigma_points(sigma_pts, accel_measured=accel)
        ukf.predict_mean_and_covariance(sigma_pts_pred)
        z_pred, S, Z_sigma = ukf.predict_measurement(sigma_pts_pred)
        ukf.update(ukf.x, ukf.P, sigma_pts_pred, z_pred, S, Z_sigma, z)

        # --- Logging ---
        ukf._update_sensor_filters(accel, gyro)
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
