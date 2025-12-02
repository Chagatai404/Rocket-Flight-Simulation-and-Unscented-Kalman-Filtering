import numpy as np
import matplotlib.pyplot as plt

# Color‑blind safe palette (Okabe–Ito)
COLORS = {
    "true_x": "#0072B2",  # blue
    "true_y": "#D55E00",  # vermillion
    "true_z": "#009E73",  # green
    "ukf_x":  "#56B4E9",  # sky blue
    "ukf_y":  "#E69F00",  # orange
    "ukf_z":  "#CC79A7",  # purple
    "true_alt": "tab:orange",
    "ukf_alt_raw": "#7A7A7A",
    "ukf_alt_smooth": "#2C7FB8"  # strong blue
}

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





def create_graphs(df, timestamps, filtered_states, filtered_accel, filtered_gyro,
                  net_body_accel_estimates, raw_accel, raw_gyro, raw_pressure):
    """
    Plot ONLY UKF estimates vs TRUE signals, plus residuals (UKF − True) and RMSE.
    Layout (10 rows): For each signal panel, a short residual panel appears underneath.
      1-2) Accel (XYZ combined) + residuals (XYZ)
      3-4) Gyro X + residual
      5-6) Gyro Y + residual
      7-8) Gyro Z + residual
      9-10) Altitude + residual
    """
    import matplotlib.gridspec as gridspec
    t = timestamps

    # Convenience: check for truth
    has_true_acc = (df is not None and all(c in df.columns for c in ['accel_true_x','accel_true_y','accel_true_z']))
    has_true_gyr = (df is not None and all(c in df.columns for c in ['gyro_true_x','gyro_true_y','gyro_true_z']))
    has_true_alt = (df is not None and 'altitude_true' in df.columns)

    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(10, 1, height_ratios=[3,1, 3,1, 3,1, 3,1, 3,1])
    ax_acc      = fig.add_subplot(gs[0, 0])
    ax_acc_res  = fig.add_subplot(gs[1, 0], sharex=ax_acc)
    ax_gx       = fig.add_subplot(gs[2, 0])
    ax_gx_res   = fig.add_subplot(gs[3, 0], sharex=ax_gx)
    ax_gy       = fig.add_subplot(gs[4, 0])
    ax_gy_res   = fig.add_subplot(gs[5, 0], sharex=ax_gy)
    ax_gz       = fig.add_subplot(gs[6, 0])
    ax_gz_res   = fig.add_subplot(gs[7, 0], sharex=ax_gz)
    ax_alt      = fig.add_subplot(gs[8, 0])
    ax_alt_res  = fig.add_subplot(gs[9, 0], sharex=ax_alt)

    # ---------- 1) ACCEL: TRUE vs UKF (combined) ----------
    if has_true_acc:
        tx = df['time'].to_numpy()
        ax_acc.plot(tx, df['accel_true_x'], label='True Accel X', color=COLORS['true_x'], linewidth=2.6)
        ax_acc.plot(tx, df['accel_true_y'], label='True Accel Y', color=COLORS['true_y'], linewidth=2.6)
        ax_acc.plot(tx, df['accel_true_z'], label='True Accel Z', color=COLORS['true_z'], linewidth=2.6)
    ax_acc.plot(t, filtered_accel[:, 0], '--', label='UKF Accel X', color=COLORS['ukf_x'], linewidth=2.0)
    ax_acc.plot(t, filtered_accel[:, 1], '--', label='UKF Accel Y', color=COLORS['ukf_y'], linewidth=2.0)
    ax_acc.plot(t, filtered_accel[:, 2], '--', label='UKF Accel Z', color=COLORS['ukf_z'], linewidth=2.0)
    ax_acc.set_title('3D Accelerometer: TRUE vs UKF')
    ax_acc.set_ylabel('m/s²')
    ax_acc.legend(ncol=3, frameon=True)
    ax_acc.grid(True)

    # Residuals for accel
    if has_true_acc:
        # Align lengths (defensive)
        n = min(len(t), len(df))
        res_x = filtered_accel[:n,0] - df['accel_true_x'].values[:n]
        res_y = filtered_accel[:n,1] - df['accel_true_y'].values[:n]
        res_z = filtered_accel[:n,2] - df['accel_true_z'].values[:n]
        ax_acc_res.plot(t[:n], res_x, color=COLORS['ukf_x'], linewidth=1.4, label=f'res X (RMSE={np.sqrt(np.mean(res_x**2)):.3f})')
        ax_acc_res.plot(t[:n], res_y, color=COLORS['ukf_y'], linewidth=1.4, label=f'res Y (RMSE={np.sqrt(np.mean(res_y**2)):.3f})')
        ax_acc_res.plot(t[:n], res_z, color=COLORS['ukf_z'], linewidth=1.4, label=f'res Z (RMSE={np.sqrt(np.mean(res_z**2)):.3f})')
        ax_acc_res.axhline(0, color='#444444', linewidth=1.0)
        ax_acc_res.set_ylabel('res')
        ax_acc_res.legend(ncol=3, fontsize=9, frameon=True)
        ax_acc_res.grid(True)

    # ---------- 2) GYRO X ----------
    if has_true_gyr:
        tx = df['time'].to_numpy()
        ax_gx.plot(tx, df['gyro_true_x'], label='True Gyro X', color=COLORS['true_x'], linewidth=2.6)
    ax_gx.plot(t, filtered_gyro[:, 0], '--', label='UKF Gyro X', color=COLORS['ukf_x'], linewidth=2.0)
    ax_gx.set_title('Gyroscope X: TRUE vs UKF')
    ax_gx.set_ylabel('rad/s')
    ax_gx.legend(frameon=True, ncol=2)
    ax_gx.grid(True)

    if has_true_gyr:
        n = min(len(t), len(df))
        res = filtered_gyro[:n,0] - df['gyro_true_x'].values[:n]
        ax_gx_res.plot(t[:n], res, color=COLORS['ukf_x'], linewidth=1.4, label=f'RMSE={np.sqrt(np.mean(res**2)):.4f}')
        ax_gx_res.axhline(0, color='#444444', linewidth=1.0)
        ax_gx_res.set_ylabel('res')
        ax_gx_res.legend(frameon=True, fontsize=9)
        ax_gx_res.grid(True)

    # ---------- 3) GYRO Y ----------
    if has_true_gyr:
        ax_gy.plot(tx, df['gyro_true_y'], label='True Gyro Y', color=COLORS['true_y'], linewidth=2.6)
    ax_gy.plot(t, filtered_gyro[:, 1], '--', label='UKF Gyro Y', color=COLORS['ukf_y'], linewidth=2.0)
    ax_gy.set_title('Gyroscope Y: TRUE vs UKF')
    ax_gy.set_ylabel('rad/s')
    ax_gy.legend(frameon=True, ncol=2)
    ax_gy.grid(True)

    if has_true_gyr:
        res = filtered_gyro[:n,1] - df['gyro_true_y'].values[:n]
        ax_gy_res.plot(t[:n], res, color=COLORS['ukf_y'], linewidth=1.4, label=f'RMSE={np.sqrt(np.mean(res**2)):.4f}')
        ax_gy_res.axhline(0, color='#444444', linewidth=1.0)
        ax_gy_res.set_ylabel('res')
        ax_gy_res.legend(frameon=True, fontsize=9)
        ax_gy_res.grid(True)

    # ---------- 4) GYRO Z ----------
    if has_true_gyr:
        ax_gz.plot(tx, df['gyro_true_z'], label='True Gyro Z', color=COLORS['true_z'], linewidth=2.6)
    ax_gz.plot(t, filtered_gyro[:, 2], '--', label='UKF Gyro Z', color=COLORS['ukf_z'], linewidth=2.0)
    ax_gz.set_title('Gyroscope Z: TRUE vs UKF')
    ax_gz.set_ylabel('rad/s')
    ax_gz.legend(frameon=True, ncol=2)
    ax_gz.grid(True)

    if has_true_gyr:
        res = filtered_gyro[:n,2] - df['gyro_true_z'].values[:n]
        ax_gz_res.plot(t[:n], res, color=COLORS['ukf_z'], linewidth=1.4, label=f'RMSE={np.sqrt(np.mean(res**2)):.4f}')
        ax_gz_res.axhline(0, color='#444444', linewidth=1.0)
        ax_gz_res.set_ylabel('res')
        ax_gz_res.legend(frameon=True, fontsize=9)
        ax_gz_res.grid(True)

    # ---------- 5) ALTITUDE ----------
    h_true = df['altitude_true'].values if has_true_alt else None
    h_ukf = filtered_states[:, 2]
    try:
        h_ukf_smooth = _butterworth_predictive_pad_smooth(t, h_ukf, fc=0.18, order=4, pad_sec=3.0, fit_sec=2.0)
    except Exception:
        h_ukf_smooth = h_ukf

    if h_true is not None:
        ax_alt.plot(df['time'], h_true, label='True Altitude (sim)', color=COLORS['true_alt'], linewidth=2.8)
    ax_alt.plot(t, h_ukf, label='UKF Estimated Altitude (raw)', color=COLORS['ukf_alt_raw'], linewidth=1.6, alpha=0.85)
    ax_alt.plot(t, h_ukf_smooth, label='UKF Altitude (smoothed)', color=COLORS['ukf_alt_smooth'], linewidth=2.4, linestyle='-')
    ax_alt.set_title('Altitude Comparison')
    ax_alt.set_ylabel('Altitude (m)')
    ax_alt.legend(frameon=True)
    ax_alt.grid(True)

    if h_true is not None:
        n = min(len(t), len(df))
        res_h = h_ukf_smooth[:n] - h_true[:n]
        ax_alt_res.plot(t[:n], res_h, color=COLORS['ukf_alt_smooth'], linewidth=1.6, label=f'RMSE={np.sqrt(np.mean(res_h**2)):.2f} m')
        ax_alt_res.axhline(0, color='#444444', linewidth=1.0)
        ax_alt_res.set_ylabel('res (m)')
        ax_alt_res.set_xlabel('Time (s)')
        ax_alt_res.legend(frameon=True, fontsize=9)
        ax_alt_res.grid(True)

    plt.tight_layout()
    plt.show()
