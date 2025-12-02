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

def create_graphs(df, timestamps, filtered_states, filtered_accel, filtered_gyro,
                  net_body_accel_estimates, raw_accel, raw_gyro, raw_pressure):
    """
    UKF vs TRUE plots without residual panels.
    Order:
      1) Accel (XYZ specific force, body)
      2) Gyro X
      3) Gyro Y  
      4) Gyro Z
      5) Altitude
    
    Note: World acceleration graph has been removed as requested.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t = timestamps
    dt = float(np.median(np.diff(t)))

    # --- availability flags for TRUE channels
    has_true_acc = (df is not None and all(c in df.columns for c in
                    ['accel_true_x','accel_true_y','accel_true_z']))
    has_true_gyr = (df is not None and all(c in df.columns for c in
                    ['gyro_true_x','gyro_true_y','gyro_true_z']))
    has_true_alt = (df is not None and 'altitude_true' in df.columns)

    # --- helpers
    def rmse(x):
        x = np.asarray(x)
        return float(np.sqrt(np.mean(x*x))) if x.size else 0.0

    def annotate_rmse(ax, text):
        ax.text(0.985, 0.985, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#333',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbb', alpha=0.95))

    def _safe_smooth(sig, fc=1.2, order=4, pad=2.0, fit=1.5):
        try:
            return _butterworth_predictive_pad_smooth(t, sig, fc=fc, order=order,
                                                      pad_sec=pad, fit_sec=fit)
        except Exception:
            return sig

    # --- vibrant palette
    COLORS = {
        'true_x': '#1f77b4',  # blue
        'true_y': '#d62728',  # red
        'true_z': '#2ca02c',  # green
        'ukf_x' : '#9467bd',  # purple
        'ukf_y' : '#ff7f0e',  # orange
        'ukf_z' : '#8c564b',  # brown
        'true_alt': '#2f4b7c',    # dark indigo
        'ukf_alt_raw': '#ff7c43', # vivid orange
        'ukf_alt_smooth': '#2ca02c' # green
    }

    # --- figure: 5 tall panels (world acceleration removed), small fonts
    with plt.rc_context({
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    }):
        fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
        gs = gridspec.GridSpec(5, 1, height_ratios=[1,1,1,1,1], figure=fig)

        ax_acc = fig.add_subplot(gs[0, 0])
        ax_gx  = fig.add_subplot(gs[1, 0])
        ax_gy  = fig.add_subplot(gs[2, 0])
        ax_gz  = fig.add_subplot(gs[3, 0])
        ax_alt = fig.add_subplot(gs[4, 0])

        # ---------- (1) ACCEL specific force (body) ----------
        acc_x_s = _safe_smooth(filtered_accel[:, 0], fc=1.2)
        acc_y_s = _safe_smooth(filtered_accel[:, 1], fc=1.2)
        acc_z_s = _safe_smooth(filtered_accel[:, 2], fc=1.2)
        accel_smoothed = np.column_stack([acc_x_s, acc_y_s, acc_z_s])

        if has_true_acc:
            tx = df['time'].to_numpy()
            ax_acc.plot(tx, df['accel_true_x'], label='True Accel X',
                        color=COLORS['true_x'], lw=2.0)
            ax_acc.plot(tx, df['accel_true_y'], label='True Accel Y',
                        color=COLORS['true_y'], lw=2.0)
            ax_acc.plot(tx, df['accel_true_z'], label='True Accel Z',
                        color=COLORS['true_z'], lw=2.0)
        ax_acc.plot(t, accel_smoothed[:, 0], '--', label='UKF Accel X',
                    color=COLORS['ukf_x'], lw=1.7)
        ax_acc.plot(t, accel_smoothed[:, 1], '--', label='UKF Accel Y',
                    color=COLORS['ukf_y'], lw=1.7)
        ax_acc.plot(t, accel_smoothed[:, 2], '--', label='UKF Accel Z',
                    color=COLORS['ukf_z'], lw=1.7)
        ax_acc.set_title('3D Accelerometer: TRUE vs UKF (specific force, body)')
        ax_acc.set_ylabel('m/s²')
        ax_acc.legend(ncol=4, frameon=True, loc='upper left')
        ax_acc.grid(True, alpha=0.3)

        if has_true_acc:
            n = min(len(t), len(df))
            rx = accel_smoothed[:n,0] - df['accel_true_x'].values[:n]
            ry = accel_smoothed[:n,1] - df['accel_true_y'].values[:n]
            rz = accel_smoothed[:n,2] - df['accel_true_z'].values[:n]
            annotate_rmse(ax_acc, f"RMSE  X={rmse(rx):.3f}  Y={rmse(ry):.3f}  Z={rmse(rz):.3f}")

        # ---------- (2) GYRO X ----------
        if has_true_gyr:
            tx = df['time'].to_numpy()
            ax_gx.plot(tx, df['gyro_true_x'], label='True Gyro X',
                       color=COLORS['true_x'], lw=2.0)
        ax_gx.plot(t, filtered_gyro[:, 0], '--', label='UKF Gyro X',
                   color=COLORS['ukf_x'], lw=1.6)
        ax_gx.set_title('Gyroscope X: TRUE vs UKF')
        ax_gx.set_ylabel('rad/s')
        ax_gx.legend(frameon=True, loc='upper left')
        ax_gx.grid(True, alpha=0.3)
        if has_true_gyr:
            n = min(len(t), len(df))
            annotate_rmse(ax_gx, f"RMSE={rmse(filtered_gyro[:n,0] - df['gyro_true_x'].values[:n]):.4f}")

        # ---------- (3) GYRO Y ----------
        if has_true_gyr:
            ax_gy.plot(tx, df['gyro_true_y'], label='True Gyro Y',
                       color=COLORS['true_y'], lw=2.0)
        ax_gy.plot(t, filtered_gyro[:, 1], '--', label='UKF Gyro Y',
                   color=COLORS['ukf_y'], lw=1.6)
        ax_gy.set_title('Gyroscope Y: TRUE vs UKF')
        ax_gy.set_ylabel('rad/s')
        ax_gy.legend(frameon=True, loc='upper left')
        ax_gy.grid(True, alpha=0.3)
        if has_true_gyr:
            annotate_rmse(ax_gy, f"RMSE={rmse(filtered_gyro[:n,1] - df['gyro_true_y'].values[:n]):.4f}")

        # ---------- (4) GYRO Z ----------
        if has_true_gyr:
            ax_gz.plot(tx, df['gyro_true_z'], label='True Gyro Z',
                       color=COLORS['true_z'], lw=2.0)
        ax_gz.plot(t, filtered_gyro[:, 2], '--', label='UKF Gyro Z',
                   color=COLORS['ukf_z'], lw=1.6)
        ax_gz.set_title('Gyroscope Z: TRUE vs UKF')
        ax_gz.set_ylabel('rad/s')
        ax_gz.legend(frameon=True, loc='upper left')
        ax_gz.grid(True, alpha=0.3)
        if has_true_gyr:
            annotate_rmse(ax_gz, f"RMSE={rmse(filtered_gyro[:n,2] - df['gyro_true_z'].values[:n]):.4f}")

        # ---------- (5) ALTITUDE ----------
        if has_true_alt:
            h_true = df['altitude_true'].values
            ax_alt.plot(df['time'], h_true, label='True Altitude (sim)',
                        color=COLORS['true_alt'], lw=2.1)
        h_ukf = filtered_states[:, 2]
        try:
            h_ukf_smooth = _butterworth_predictive_pad_smooth(t, h_ukf, fc=0.18, order=4, pad_sec=3.0, fit_sec=2.0)
        except Exception:
            h_ukf_smooth = h_ukf
        ax_alt.plot(t, h_ukf,        label='UKF Altitude (raw)',
                    color=COLORS['ukf_alt_raw'], lw=1.0, alpha=0.85)
        ax_alt.plot(t, h_ukf_smooth, label='UKF Altitude (smoothed)',
                    color=COLORS['ukf_alt_smooth'], lw=1.9)
        ax_alt.set_title('Altitude Comparison')
        ax_alt.set_ylabel('Altitude (m)')
        ax_alt.set_xlabel('Time (s)')
        ax_alt.legend(ncol=3, frameon=True, loc='upper left')
        ax_alt.grid(True, alpha=0.3)
        if has_true_alt:
            n = min(len(t), len(df))
            annotate_rmse(ax_alt, f"RMSE={rmse(h_ukf_smooth[:n] - h_true[:n]):.2f} m")

        # hide repeated x labels to reduce clutter
        for ax in [ax_acc, ax_gx, ax_gy, ax_gz]:
            ax.label_outer()

        plt.show()