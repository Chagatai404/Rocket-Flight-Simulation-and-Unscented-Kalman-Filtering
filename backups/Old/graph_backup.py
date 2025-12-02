import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    axs[3].plot(t, filtered_states[:, 2], label='UKF Estimated Altitude (z)', color='navy')
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
