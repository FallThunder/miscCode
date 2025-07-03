import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 3.0                 # Navigation constant
Vm = 400.0              # Missile speed (m/s)
Vt = 250.0              # Target speed (m/s)
dt = 0.005              # Time step (s)
T_total = 30            # Total simulation time (s)
max_turn_rate = np.deg2rad(15)  # Max turn rate (rad/s)

# Initial positions (x, y, z)
R_m0 = np.array([0.0, 0.0, 0.0])       # Missile starts at origin, altitude 0 m
R_t0 = np.array([3000.0, 0.0, 500.0])  # Target starts 3 km ahead, 500 m altitude

# Initial headings (azimuth and elevation angles in radians)
# Azimuth angle: angle in x-y plane from x-axis
# Elevation angle: angle from horizontal plane upward

gamma_m_az = 0.0                   # Missile azimuth heading
gamma_m_el = 0.0                   # Missile elevation heading

gamma_t_az = np.deg2rad(10)        # Target azimuth 10 degrees
gamma_t_el = np.deg2rad(5)         # Target elevation 5 degrees

def direction_vector(azimuth, elevation):
    """Convert azimuth and elevation angles to 3D unit direction vector."""
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.array([x, y, z])

def simulate_pn_3d():
    t = 0.0
    R_m = R_m0.copy()
    R_t = R_t0.copy()
    gamma_m_az_current = gamma_m_az
    gamma_m_el_current = gamma_m_el

    missile_pos = []
    target_pos = []

    while t < T_total:
        # Relative position and range
        R = R_t - R_m
        R_norm = np.linalg.norm(R)

        # LOS unit vector
        los = R / R_norm

        # Missile and target velocity vectors
        V_m_dir = direction_vector(gamma_m_az_current, gamma_m_el_current)
        V_m_vec = Vm * V_m_dir

        V_t_dir = direction_vector(gamma_t_az, gamma_t_el)
        V_t_vec = Vt * V_t_dir

        V_r = V_t_vec - V_m_vec

        # LOS rate vector: Omega = (R x V_r) / |R|^2
        Omega = np.cross(R, V_r) / (R_norm**2)

        # Magnitude of LOS rate vector
        omega_mag = np.linalg.norm(Omega)

        # Closing velocity Vc = -dR/dt = - (R · V_r) / |R|
        dR_dt = np.dot(R, V_r) / R_norm
        V_c = -dR_dt

        # Commanded lateral acceleration magnitude
        a_n = N * V_c * omega_mag

        if omega_mag > 1e-6:
            # Direction of lateral acceleration is perpendicular to LOS and in plane of rotation
            a_n_dir = -np.cross(los, Omega / omega_mag)
        else:
            a_n_dir = np.array([0.0, 0.0, 0.0])

        # Acceleration vector
        a_n_vec = a_n * a_n_dir

        # Calculate missile heading change rate vector: gamma_dot = a_n_vec / Vm
        gamma_dot_vec = a_n_vec / Vm

        # Convert gamma_dot_vec to azimuth and elevation rate (approximate)
        # This is a simplification: update azimuth and elevation angles by projecting gamma_dot_vec

        # Azimuth change: project gamma_dot_vec onto horizontal plane normal to missile velocity
        hor_proj = gamma_dot_vec - np.dot(gamma_dot_vec, V_m_dir) * V_m_dir
        hor_proj_norm = np.linalg.norm(hor_proj)
        if hor_proj_norm > 1e-8:
            # Compute azimuth and elevation changes
            # Approximate azimuth rate from horizontal projection
            az_dot = hor_proj[1] / np.cos(gamma_m_el_current)  # y-component / cos(elevation)
            el_dot = gamma_dot_vec[2]  # z-component as elevation rate
        else:
            az_dot = 0.0
            el_dot = 0.0

        # Limit turn rates (simple scalar clipping)
        az_dot = np.clip(az_dot, -max_turn_rate, max_turn_rate)
        el_dot = np.clip(el_dot, -max_turn_rate, max_turn_rate)

        # Update missile headings
        gamma_m_az_current += az_dot * dt
        gamma_m_el_current += el_dot * dt

        # Update positions
        R_m += Vm * direction_vector(gamma_m_az_current, gamma_m_el_current) * dt
        R_t += Vt * V_t_dir * dt

        missile_pos.append(R_m.copy())
        target_pos.append(R_t.copy())

        t += dt

        # Stop if missile close to target
        if R_norm < 10:
            print(f"Target intercepted at time {t:.2f} seconds!")
            break

    return np.array(missile_pos), np.array(target_pos)

missile_pos, target_pos = simulate_pn_3d()

# Plotting 3D trajectories
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(missile_pos[:,0], missile_pos[:,1], missile_pos[:,2], label='Missile trajectory')
ax.plot(target_pos[:,0], target_pos[:,1], target_pos[:,2], label='Target trajectory')

ax.scatter(missile_pos[0,0], missile_pos[0,1], missile_pos[0,2], c='green', label='Missile start')
ax.scatter(target_pos[0,0], target_pos[0,1], target_pos[0,2], c='red', label='Target start')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Proportional Navigation Missile Guidance Simulation')
ax.legend()
plt.show()
