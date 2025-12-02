import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt


def _butterworth_predictive_pad_smooth(t, y, fc=0.2, order=4, pad_sec=3.0, fit_sec=2.0):
    """
    Smooth y(t) with a zero-phase Butterworth filter using predictive parabolic padding at both ends.
    This avoids tail artifacts seen with reflection padding while keeping the curve very smooth.

    Parameters:
        t (np.ndarray): time array (N,)
        y (np.ndarray): signal array (N,)
        fc (float): low-pass cutoff in Hz (default 0.2)
        order (int): Butterworth order (default 4)
        pad_sec (float): padding length on each side in seconds (default 3.0)
        fit_sec (float): duration used to fit end parabolas (default 2.0)

    Returns:
        np.ndarray: smoothed signal (N,)
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    wn = float(fc) / (fs / 2.0)

    # Butterworth design
    b, a = butter(N=order, Wn=wn, btype="low", analog=False)

    padN = max(10, int(round(pad_sec / dt)))
    fitN = max(10, int(round(fit_sec / dt)))

    # Left parabola fit
    coeff_left = np.polyfit(t[:fitN], y[:fitN], deg=2)
    t_left = t[0] - np.arange(padN, 0, -1) * dt
    y_left = np.polyval(coeff_left, t_left)

    # Right parabola fit
    coeff_right = np.polyfit(t[-fitN:], y[-fitN:], deg=2)
    t_right = t[-1] + np.arange(1, padN + 1) * dt
    y_right = np.polyval(coeff_right, t_right)

    x_ext = np.concatenate([y_left, y, y_right])
    y_ext = filtfilt(b, a, x_ext, method="pad")
    return y_ext[padN:-padN]


def create_graphs(
    df, timestamps, filtered_states,
    filtered_accel, filtered_gyro,
    net_body_accel_estimates,
    raw_accel, raw_gyro, raw_pressure
):
    """
    Create visualization plots for:
      - Accelerometer (raw vs. model gravity-only)
      - Gyroscope (raw + UKF)
      - Barometric pressure
      - UKF estimated altitude (z)
      - Net body acceleration (from prediction step)

    Parameters:
        df (pd.DataFrame): dataframe used for truth references if needed.
        timestamps (np.ndarray): time axis (N,).
        filtered_states (np.ndarray): UKF state history (N, 13).
        filtered_accel (np.ndarray): gravity-only accel from UKF state (N,3).
        filtered_gyro (np.ndarray): UKF-estimated angular rates (N,3).
        net_body_accel_estimates (np.ndarray): body accel used in prediction (N,3).
        model_accel (np.ndarray): gravity-only model accel (N,3).
        raw_accel (np.ndarray): raw accelerometer samples (N,3).
        raw_gyro (np.ndarray): raw gyro samples (N,3).
        raw_pressure (np.ndarray): raw pressure samples (N,).
    """
    t = timestamps

    fig, axs = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

    # 1) Accelerometer: Raw vs Model (gravity-only)
    #    Raw shows what the IMU measured; model shows what the UKF predicts if only gravity is present.
    axs[0].plot(t, raw_accel[:, 0], label='Raw Accel X', color='red', alpha=0.35)
    axs[0].plot(t, raw_accel[:, 1], label='Raw Accel Y', color='green', alpha=0.35)
    axs[0].plot(t, raw_accel[:, 2], label='Raw Accel Z', color='blue', alpha=0.35)
    axs[0].plot(t, filtered_accel[:, 0], '--', label='UKF Accel X', color='red')
    axs[0].plot(t, filtered_accel[:, 1], '--', label='UKF Accel Y', color='green')
    axs[0].plot(t, filtered_accel[:, 2], '--', label='UKF Accel Z', color='blue')
    axs[0].set_title('3D Accelerometer: Raw vs UKF-Filtered')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend(ncol=2)
    axs[0].grid(True)

    # 2) Gyroscope: Raw + UKF-estimated (from state)
    axs[1].plot(t, raw_gyro[:, 0], label='Raw Gyro X', color='red', alpha=0.35)
    axs[1].plot(t, raw_gyro[:, 1], label='Raw Gyro Y', color='green', alpha=0.35)
    axs[1].plot(t, raw_gyro[:, 2], label='Raw Gyro Z', color='blue', alpha=0.35)
    axs[1].plot(t, filtered_gyro[:, 0], label='UKF Gyro X', color='red')
    axs[1].plot(t, filtered_gyro[:, 1], label='UKF Gyro Y', color='green')
    axs[1].plot(t, filtered_gyro[:, 2], label='UKF Gyro Z', color='blue')
    axs[1].set_title('3D Gyroscope (Raw + UKF)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend(ncol=2)
    axs[1].grid(True)

    # 3) Barometric Pressure (raw)
    axs[2].plot(t, raw_pressure, label='Pressure (Measured)', color='purple')
    axs[2].set_title('Barometric Pressure (Noisy)')
    axs[2].set_ylabel('Pressure (hPa)')
    axs[2].legend()
    axs[2].grid(True)

    # 4) UKF Estimated Altitude (z position from state)
    # Altitude panel shows: true altitude (if available), raw UKF altitude,
    # and a presentation‑quality smooth curve produced by Butterworth low‑pass
    # filtering with predictive parabolic padding to avoid end artifacts.
    est_alt = filtered_states[:, 2]
    # Predictive padded Butterworth smoothing (presentation curve)
    try:
        est_alt_smooth = _butterworth_predictive_pad_smooth(t, est_alt, fc=0.2)
    except Exception:
        est_alt_smooth = est_alt
    if df is not None and 'altitude_true' in df.columns:
        axs[3].plot(df['time'].to_numpy(), df['altitude_true'].to_numpy(), label='True Altitude (sim)', linewidth=2.5, color='tab:orange')
    axs[3].plot(t, est_alt, label='UKF Estimated Altitude (raw)', color='navy', alpha=0.35)
    axs[3].plot(t, est_alt_smooth, label='UKF Altitude (Butterworth + predictive padding)', linewidth=2)
    axs[3].set_title('UKF Estimated Altitude Over Time')
    axs[3].set_ylabel('Altitude (m)')
    axs[3].legend()
    axs[3].grid(True)

    # 5) Net Body Acceleration (from prediction step)
    #    This is the acceleration in the body frame that the UKF actually used (from the IMU),
    #    so it should track thrust/drag/etc. and can show spikes from the sensor.
    axs[4].plot(t, net_body_accel_estimates[:, 0], '--', label='Net Body Accel X', color='red')
    axs[4].plot(t, net_body_accel_estimates[:, 1], '--', label='Net Body Accel Y', color='green')
    axs[4].plot(t, net_body_accel_estimates[:, 2], '--', label='Net Body Accel Z', color='blue')
    axs[4].set_title('Estimated Net Body Acceleration (UKF Prediction Input)')
    axs[4].set_ylabel('Acceleration (m/s²)')
    axs[4].legend(ncol=2)
    axs[4].grid(True)

    # 6) Spacer / x-axis
    axs[5].set_xlabel('Time (s)')
    axs[5].axis('off')

    plt.tight_layout()
    plt.show()
