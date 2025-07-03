import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 3.0                 # Navigation constant
Vm = 400.0              # Missile speed (m/s)
Vt = 250.0              # Target speed (m/s)
dt = 0.005              # Time step (s)
T_total = 30            # Total simulation time (s)
max_turn_rate = np.deg2rad(15)  # Max turn rate (rad/s)

# Target randomness parameters
maneuver_interval = 2.0         # Time between random maneuvers (s)
max_heading_change = np.deg2rad(30)  # Max random heading change (rad)
velocity_noise_factor = 0.1     # Velocity variation factor
max_target_turn_rate = np.deg2rad(10)  # Max target turn rate (rad/s)

# Base initial positions (will be randomized in simulation)
R_m0_base = np.array([0.0, 0.0, 0.0])       # Missile base position
R_t0_base = np.array([3000.0, 0.0, 500.0])  # Target base position

# Randomization ranges for initial positions
missile_pos_range = 200.0    # Missile can start within ±200m of base position
target_pos_range = 500.0     # Target can start within ±500m of base position
target_altitude_range = 300.0  # Target altitude can vary by ±300m

# Base initial headings (azimuth and elevation angles in radians)
# Azimuth angle: angle in x-y plane from x-axis
# Elevation angle: angle from horizontal plane upward

gamma_m_az_base = 0.0                   # Missile base azimuth heading
gamma_m_el_base = 0.0                   # Missile base elevation heading

gamma_t_az_base = np.deg2rad(10)        # Target base azimuth 10 degrees
gamma_t_el_base = np.deg2rad(5)         # Target base elevation 5 degrees

# Randomization ranges for initial headings
initial_heading_range = np.deg2rad(15)  # ±15 degrees variation in initial headings

def direction_vector(azimuth, elevation):
    """Convert azimuth and elevation angles to 3D unit direction vector."""
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.array([x, y, z])

def simulate_pn_3d():
    t = 0.0
    
    # Random seed for different results each run
    np.random.seed()
    
    # Generate random starting positions
    R_m0 = R_m0_base + np.random.uniform(-missile_pos_range, missile_pos_range, 3)
    R_m0[2] = max(0.0, R_m0[2])  # Keep missile altitude non-negative
    
    R_t0 = R_t0_base.copy()
    R_t0[0] += np.random.uniform(-target_pos_range, target_pos_range)  # X position
    R_t0[1] += np.random.uniform(-target_pos_range, target_pos_range)  # Y position
    R_t0[2] += np.random.uniform(-target_altitude_range, target_altitude_range)  # Z position
    R_t0[2] = max(50.0, R_t0[2])  # Keep target altitude at least 50m
    
    print(f"Missile start position: ({R_m0[0]:.1f}, {R_m0[1]:.1f}, {R_m0[2]:.1f}) m")
    print(f"Target start position: ({R_t0[0]:.1f}, {R_t0[1]:.1f}, {R_t0[2]:.1f}) m")
    print(f"Initial separation: {np.linalg.norm(R_t0 - R_m0):.1f} m")
    
    # Generate random initial headings
    gamma_m_az = gamma_m_az_base + np.random.uniform(-initial_heading_range, initial_heading_range)
    gamma_m_el = gamma_m_el_base + np.random.uniform(-initial_heading_range/2, initial_heading_range/2)
    
    gamma_t_az = gamma_t_az_base + np.random.uniform(-initial_heading_range, initial_heading_range)
    gamma_t_el = gamma_t_el_base + np.random.uniform(-initial_heading_range/2, initial_heading_range/2)
    
    print(f"Missile initial heading: Az={np.rad2deg(gamma_m_az):.1f}°, El={np.rad2deg(gamma_m_el):.1f}°")
    print(f"Target initial heading: Az={np.rad2deg(gamma_t_az):.1f}°, El={np.rad2deg(gamma_t_el):.1f}°")
    
    R_m = R_m0.copy()
    R_t = R_t0.copy()
    gamma_m_az_current = gamma_m_az
    gamma_m_el_current = gamma_m_el
    
    # Target dynamic parameters
    gamma_t_az_current = gamma_t_az
    gamma_t_el_current = gamma_t_el
    Vt_current = Vt
    last_maneuver_time = 0.0

    missile_pos = []
    target_pos = []

    while t < T_total:
        # Relative position and range
        R = R_t - R_m
        R_norm = np.linalg.norm(R)

        # LOS unit vector
        los = R / R_norm

        # Target evasive maneuvers and random behavior
        if t - last_maneuver_time >= maneuver_interval:
            # Random heading changes
            gamma_t_az_current += np.random.uniform(-max_heading_change, max_heading_change)
            gamma_t_el_current += np.random.uniform(-max_heading_change/2, max_heading_change/2)
            
            # Random velocity changes
            Vt_current = Vt * (1 + np.random.uniform(-velocity_noise_factor, velocity_noise_factor))
            
            last_maneuver_time = t
            print(f"Target maneuver at t={t:.2f}s: Az={np.rad2deg(gamma_t_az_current):.1f}°, El={np.rad2deg(gamma_t_el_current):.1f}°, V={Vt_current:.1f}m/s")
        
        # Continuous small random perturbations to target heading
        gamma_t_az_current += np.random.uniform(-max_target_turn_rate*dt, max_target_turn_rate*dt)
        gamma_t_el_current += np.random.uniform(-max_target_turn_rate*dt/2, max_target_turn_rate*dt/2)
        
        # Clamp elevation to reasonable values
        gamma_t_el_current = np.clip(gamma_t_el_current, -np.deg2rad(45), np.deg2rad(45))

        # Missile and target velocity vectors
        V_m_dir = direction_vector(gamma_m_az_current, gamma_m_el_current)
        V_m_vec = Vm * V_m_dir

        V_t_dir = direction_vector(gamma_t_az_current, gamma_t_el_current)
        V_t_vec = Vt_current * V_t_dir

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
        R_t += Vt_current * V_t_dir * dt

        missile_pos.append(R_m.copy())
        target_pos.append(R_t.copy())

        t += dt

        # Stop if missile close to target
        if R_norm < 10:
            print(f"Target intercepted at time {t:.2f} seconds!")
            break

    if t >= T_total:
        print(f"Simulation ended at {T_total} seconds. Final range: {R_norm:.1f} meters")
    
    return np.array(missile_pos), np.array(target_pos)

missile_pos, target_pos = simulate_pn_3d()

# Plotting 3D trajectories
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories with varying colors to show progression
n_points = len(missile_pos)
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Plot missile trajectory
ax.plot(missile_pos[:,0], missile_pos[:,1], missile_pos[:,2], 
        color='blue', linewidth=2, label='Missile trajectory')

# Plot target trajectory with color gradient to show evasive maneuvers
ax.plot(target_pos[:,0], target_pos[:,1], target_pos[:,2], 
        color='red', linewidth=2, label='Target trajectory (evasive)')

# Mark start and end positions
ax.scatter(missile_pos[0,0], missile_pos[0,1], missile_pos[0,2], 
          c='green', s=100, label='Missile start', marker='o')
ax.scatter(target_pos[0,0], target_pos[0,1], target_pos[0,2], 
          c='red', s=100, label='Target start', marker='s')

# Mark final positions
ax.scatter(missile_pos[-1,0], missile_pos[-1,1], missile_pos[-1,2], 
          c='blue', s=100, label='Missile end', marker='x')
ax.scatter(target_pos[-1,0], target_pos[-1,1], target_pos[-1,2], 
          c='darkred', s=100, label='Target end', marker='x')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Proportional Navigation with Evasive Target\n(Random start positions, headings, and maneuvers every 2 seconds)')
ax.legend()
ax.grid(True)

# Set equal aspect ratio
max_range = np.array([missile_pos[:,0].max()-missile_pos[:,0].min(),
                      missile_pos[:,1].max()-missile_pos[:,1].min(),
                      missile_pos[:,2].max()-missile_pos[:,2].min()]).max() / 2.0
mid_x = (missile_pos[:,0].max()+missile_pos[:,0].min()) * 0.5
mid_y = (missile_pos[:,1].max()+missile_pos[:,1].min()) * 0.5
mid_z = (missile_pos[:,2].max()+missile_pos[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()
