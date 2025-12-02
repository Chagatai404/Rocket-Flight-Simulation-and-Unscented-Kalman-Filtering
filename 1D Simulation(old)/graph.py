import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_graphs(data):
    
    # Estimate altitude from barometric pressure (inverse ISA)
    P = data['pressure_measured']
    P0 = 1013.25
    T0 = 288.15
    L = 0.0065
    R = 8.31447
    M = 0.0289644
    g0 = 9.80665

    # Avoid invalid values (e.g., pressure < 0)
    P = np.clip(P, 1, P0)

    # Calculate estimated altitude from pressure
    altitude_from_pressure = T0 / L * (1 - (P / P0) ** ((R * L) / (g0 * M)))

    # === Plot ===
    plt.figure(figsize=(15, 12))

    # Altitude: true vs. from pressure
    plt.subplot(3, 1, 1)
    plt.plot(data['time'], data['altitude_true'], label='True Altitude (m)', color='blue')
    plt.plot(data['time'], altitude_from_pressure, label='Estimated Altitude (from pressure)', color='skyblue', linestyle='--')
    plt.title('Altitude: True vs. Estimated from Pressure')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.legend()

    # Acceleration: true vs. noisy
    plt.subplot(3, 1, 2)
    plt.plot(data['time'], data['acceleration_true'], label='True Acceleration (m/s²)', color='red')
    plt.plot(data['time'], data['accel_measured'], label='Accelerometer Reading (noisy)', color='orange', linestyle='--')
    plt.title('Acceleration: True vs. Accelerometer')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid()
    plt.legend()

    # Gyroscope: noisy only
    plt.subplot(3, 1, 3)
    plt.plot(data['time'], data['gyro_measured'], label='Gyroscope Reading (noisy)', color='purple')
    plt.title('Gyroscope (no rotation assumed)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

