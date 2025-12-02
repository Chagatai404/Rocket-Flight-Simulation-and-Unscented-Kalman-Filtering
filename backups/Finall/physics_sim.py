# Simulate a 1D rocket flight with full 3D IMU (accelerometer + gyro) and quaternion orientation
import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R

def simulate_rocket(T_total=50, dt=0.01, save=False, save_path="datasets/rocket_flight_data.csv",
                    target_peak_g=8.83, target_apogee_m=2920.0, auto_calibrate=True):
    # === Constants for physics ===
    g = 9.81
    rho = 1.225
    # Aerodynamics (sleek 141 mm airframe; slightly optimistic Cd for competition rockets)
    Cd_rocket = 0.35
    A_rocket = np.pi * (0.141/2)**2      # 141 mm dia -> ~0.0156 m^2
    Cd_chute = 1.5
    A_chute = 0.8                        # tune if you want specific descent rate

    # === Vehicle masses (from your table; grams -> kg) ===
    m0 = 24.945        # total liftoff mass (kg)
    mass_loss = 4.122  # propellant mass (kg)

    # === Motor timing ===
    burn_time = 2.8    # s  (M1850-like)
    ramp_time = 0.25   # s  (tail-off smoothing)

    # --- Thrust levels set by targets ---
    # Peak-g target -> plateau thrust needed at liftoff (drag ~0): T_plateau = m*(g + g*target_peak_g)
    T_plateau = m0 * g * (1.0 + target_peak_g)                # ~2405 N for 8.83 g
    T_peak    = 1.08 * T_plateau                              # small headroom over plateau
    # Shape: quick rise, sustain, fall (rough M1850 character)
    t_rise = 0.10 * burn_time
    t_hold = 0.60 * burn_time
    t_fall = burn_time - (t_rise + t_hold)

    def thrust_profile_base(t):
        if t <= 0.0:
            return 0.0
        if t <= t_rise:
            return (T_peak / t_rise) * t
        elif t <= t_rise + t_hold:
            return T_plateau
        elif t <= burn_time:
            # linear fall from plateau to ~0 at burn_time (tail controlled by ramp_time)
            return np.interp(t, [t_rise + t_hold, burn_time], [T_plateau, 0.0])
        else:
            return 0.0

    # ---- Simple inner simulator for apogee (no noise, no chute) ----
    def simulate_apogee(thrust_scale):
        n = int(T_total/dt)
        v = 0.0
        h = 0.0
        m = m0
        # mass timeline (linear burn)
        dry_mass = m0 - mass_loss
        for i in range(1, n):
            t = i*dt
            # update mass
            if t <= burn_time:
                m = m0 - mass_loss * (t / burn_time)
            else:
                m = dry_mass
            # thrust with scale + tail ramp
            T = thrust_scale * thrust_profile_base(t)
            if (t > burn_time) and (t <= burn_time + ramp_time):
                T *= 1 - (t - burn_time)/ramp_time
            elif t > burn_time + ramp_time:
                T = 0.0
            # aero (rocket only)
            CdA = Cd_rocket * A_rocket
            F_drag = 0.5 * rho * CdA * v*v * np.sign(v)
            a = (T - F_drag - m*g)/m
            v += a*dt
            h += v*dt
            if i > 5 and v <= 0 and h > 0:  # apogee reached
                break
        return max(h, 0.0)

    # --- Auto-calibrate thrust scale for target apogee ---
    thrust_scale = 1.0
    if auto_calibrate:
        # bracket scales
        low, high = 0.6, 2.0
        h_low  = simulate_apogee(low)
        h_high = simulate_apogee(high)
        # widen if needed
        iters = 0
        while h_high < target_apogee_m and iters < 6:
            high *= 1.5
            h_high = simulate_apogee(high)
            iters += 1
        # bisection to ~1% height error (max 18 iterations)
        for _ in range(18):
            mid = 0.5*(low+high)
            h_mid = simulate_apogee(mid)
            if abs(h_mid - target_apogee_m) <= 0.01*target_apogee_m:
                thrust_scale = mid
                break
            if h_mid < target_apogee_m:
                low, h_low = mid, h_mid
            else:
                high, h_high = mid, h_mid
            thrust_scale = 0.5*(low+high)

    # === Parachute deployment settings ===
    deploy_time = None
    deploy_on_apogee = True
    deploy_delay = 0.2          # seconds after apogee
    chute_open_time = 0.4       # seconds to fully inflate
    ejection_impulse_Ns = 25.0  # N*s impulse
    ejection_force_dt = 0.02   # 20 ms force pulse

    # === Time array ===
    n = int(T_total / dt)
    time = np.linspace(0, T_total, n)

    # === Initialize state arrays ===
    h = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    mass = np.full(n, m0)

    # Mass burn then hold dry mass (bug-fixed)
    for i in range(1, int(burn_time/dt)):
        mass[i] = m0 - mass_loss * (i*dt / burn_time)
    mass[int(burn_time/dt):] = mass[int(burn_time/dt)-1]

    # Deploy state
    deployed = False
    deploy_start_t = None
    apogee_seen = False

    # === Physics loop ===
    for i in range(1, n):
        v_prev = v[i-1]
        m = mass[i]

        # --- 1) predict for apogee detection using rocket-only CdA
        CdA_rocket = Cd_rocket * A_rocket
        F_drag_rocket = 0.5 * rho * CdA_rocket * v_prev*v_prev * np.sign(v_prev)
        # thrust with scale + tail
        F_thrust = thrust_scale * thrust_profile_base(time[i])
        if (time[i] > burn_time) and (time[i] <= burn_time + ramp_time):
            F_thrust *= 1 - (time[i] - burn_time)/ramp_time
        elif time[i] > burn_time + ramp_time:
            F_thrust = 0.0
        a_pred = (F_thrust - F_drag_rocket - m*g)/m
        v_pred = v_prev + a_pred*dt

        if deploy_on_apogee and not apogee_seen and v_prev > 0 and v_pred <= 0:
            apogee_seen = True
            frac = v_prev / (v_prev - v_pred)   # 0..1 in-step zero crossing
            t_apogee = time[i-1] + frac*dt
            deploy_start_t = t_apogee + deploy_delay

        if (deploy_time is not None) and (deploy_start_t is None) and (time[i] >= deploy_time):
            deploy_start_t = time[i] + deploy_delay

        # --- 2) apply ejection impulse at deploy time
        if (not deployed) and (deploy_start_t is not None) and (time[i] >= deploy_start_t):
            deployed = True
            if abs(v_prev) > 1e-6:
                v_prev += -(ejection_impulse_Ns / m) * np.sign(v_prev)

        # --- 3) final forces with correct CdA (chute ramp when deployed)
        if deployed:
            tau = np.clip((time[i] - deploy_start_t)/chute_open_time, 0.0, 1.0)
            CdA_current = (1.0 - tau)*(Cd_rocket*A_rocket) + tau*(Cd_chute*A_chute)
        else:
            CdA_current = Cd_rocket*A_rocket

        F_drag = 0.5 * rho * CdA_current * v_prev*v_prev * np.sign(v_prev)

        # recompute thrust explicitly for final step
        F_thrust = thrust_scale * thrust_profile_base(time[i])
        if (time[i] > burn_time) and (time[i] <= burn_time + ramp_time):
            F_thrust *= 1 - (time[i] - burn_time)/ramp_time
        elif time[i] > burn_time + ramp_time:
            F_thrust = 0.0

        # Extra force during the ejection pulse window
        F_eject = 0.0
        if deployed and (time[i] < deploy_start_t + ejection_force_dt):
            # convert impulse (N*s) to force (N) over the short window
            F_eject = (ejection_impulse_Ns / ejection_force_dt) * (-np.sign(v_prev or 1.0))
            # if v_prev is exactly 0, use -1.0 or the last nonzero sign

        net_force = F_thrust - F_drag - m*g + F_eject

        a[i] = net_force / m
        v[i] = v_prev + a[i]*dt
        h[i] = max(0.0, h[i-1] + v[i]*dt)

    # === Quaternion orientation (synthetic), IMU specific-force, gyro, and sensors ===
    angular_rates = np.deg2rad(np.vstack([
        2 * np.sin(0.2 * time),
        1.5 * np.cos(0.1 * time),
        1.0 * np.sin(0.3 * time)
    ])).T

    quaternions = []
    r = R.identity()
    for i in range(n):
        rotvec = angular_rates[i]*dt
        r = r * R.from_rotvec(rotvec)
        quaternions.append(r.as_quat())  # [x,y,z,w]
    quaternions = np.array(quaternions)
    q_x, q_y, q_z, q_w = quaternions[:,0], quaternions[:,1], quaternions[:,2], quaternions[:,3]

    # Specific force: f_b = R_wb (a_world + g_world)
    acc_world = np.zeros((n,3))
    acc_world[:,2] = a
    gravity_world = np.array([0,0,-g])
    acc_body = np.zeros_like(acc_world)
    for i in range(n):
        r = R.from_quat(quaternions[i])
        acc_body[i] = r.inv().apply(acc_world[i] + gravity_world)

    # Gyro from quaternion diffs
    gyro_body = np.zeros((n,3))
    for i in range(1,n):
        r_prev = R.from_quat(quaternions[i-1])
        r_curr = R.from_quat(quaternions[i])
        delta_r = r_curr * r_prev.inv()
        gyro_body[i] = delta_r.as_rotvec()/dt

    # Sensor noise / drift / outliers
    def generate_bias_drift(n, bias_mean, bias_std, drift_rate):
        base_bias = np.random.normal(bias_mean, bias_std, size=(n,1))
        drift = np.linspace(0, drift_rate, n).reshape(-1,1)
        return base_bias + drift

    def inject_outliers(signal, num_outliers=20, magnitude=10.0):
        for dim in range(signal.shape[1]):
            idx = np.random.choice(signal.shape[0], size=num_outliers, replace=False)
            signal[idx, dim] += np.random.normal(0, magnitude, size=num_outliers)
        return signal

    accel_noise_std = 1.0
    gyro_noise_std  = 0.05
    accel_drift = generate_bias_drift(n, 0.5, 0.2, 0.5)
    gyro_drift  = generate_bias_drift(n, 0.01, 0.01, 0.02)

    accel_measured = acc_body + np.random.normal(0, accel_noise_std, acc_body.shape) + accel_drift
    gyro_measured  = gyro_body  + np.random.normal(0, gyro_noise_std,  gyro_body.shape)  + gyro_drift

    accel_measured = inject_outliers(accel_measured, num_outliers=20, magnitude=10)
    gyro_measured  = inject_outliers(gyro_measured,  num_outliers=20, magnitude=0.1)

    # Baro
    P0, T0, L = 1013.25, 288.15, 0.0065
    R_gas, M, g0 = 8.31447, 0.0289644, 9.80665
    pressure_true = P0 * (1 - (L*h)/T0) ** ((g0*M)/(R_gas*L))
    pressure_measured = pressure_true + np.random.normal(0, 5.0, size=n)

    # Output DataFrame (schema unchanged)
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
        'accel_x': accel_measured[:,0],
        'accel_y': accel_measured[:,1],
        'accel_z': accel_measured[:,2],
        'gyro_x': gyro_measured[:,0],
        'gyro_y': gyro_measured[:,1],
        'gyro_z': gyro_measured[:,2],
        'pressure_measured': pressure_measured,
        'q_x': q_x, 'q_y': q_y, 'q_z': q_z, 'q_w': q_w
    })

    if save:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        clean_cols = ['time','altitude_true','velocity_true','acceleration_true','mass','pressure_true',
                      'accel_true_x','accel_true_y','accel_true_z',
                      'gyro_true_x','gyro_true_y','gyro_true_z','q_x','q_y','q_z','q_w']
        data[clean_cols].to_csv(save_path.replace(".csv","_clean.csv"), index=False)
        data.to_csv(save_path.replace(".csv","_noisy.csv"), index=False)

    return data
