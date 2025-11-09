import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
from tqdm import tqdm
import sys
import os
import matplotlib as mpl


# --- 1. 3D Physical Model ---
def model_3d(t, state, m, c, g, wind_vec):
    """
    Defines the system of differential equations for 3D projectile motion
    Coordinate system: X=forward, Y=lateral deviation, Z=vertical height
    state = [x, y, z, vx, vy, vz]
    wind_vec = [wx, wy, wz]
    """
    # Unpack state variables (velocity)
    vx, vy, vz = state[3], state[4], state[5]
    
    # Unpack wind components
    wx, wy, wz = wind_vec
    
    # 1. Calculate relative velocity
    v_rel_x = vx - wx
    v_rel_y = vy - wy
    v_rel_z = vz - wz
    
    # 2. Calculate relative speed (magnitude)
    v_rel = (v_rel_x**2 + v_rel_y**2 + v_rel_z**2)**0.5
    
    if v_rel < 1e-9:
        # Special case: projectile stationary relative to air
        ax = 0
        ay = 0
        az = -g  # Gravity in Z axis
    else:
        # 3. Calculate 3D air resistance (F_d = -c * v_rel * v_rel_vec)
        # 4. Calculate 3D acceleration (a = F_net / m)
        ax = -(c / m) * v_rel * v_rel_x
        ay = -(c / m) * v_rel * v_rel_y
        az = -g - (c / m) * v_rel * v_rel_z  # Gravity in Z axis
    
    # Return derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    return [vx, vy, vz, ax, ay, az]

# --- 2. Objective Function ---
def objective_function_3d(launch_angles, v0, m, c, g, wind_vec, target_x, target_z):
    """
    Given a set of [theta, phi], simulate the trajectory,
    and return [y_error, z_error] at x=target_x
    """
    
    theta_deg, phi_deg = launch_angles
    
    theta_rad = np.radians(theta_deg)  # Elevation angle (X-Z plane)
    phi_rad = np.radians(phi_deg)      # Azimuth angle (X-Y plane)
    
    # 1. Calculate 3D initial velocity 
    # Vertical velocity (Z axis)
    v0z = v0 * np.sin(theta_rad)
    # Projection in X-Y horizontal plane
    v_proj_xy = v0 * np.cos(theta_rad)
    
    # Decompose horizontal projection
    v0x = v_proj_xy * np.cos(phi_rad)  # Forward direction (X axis)
    v0y = v_proj_xy * np.sin(phi_rad)  # Lateral deviation (Y axis)
    
    initial_state = [0, 0, 0, v0x, v0y, v0z]  # [x, y, z, vx, vy, vz]
    
    # 2. Define stop event: when x coordinate = target_x
    def stop_at_target_x(t, state, m, c, g, wind_vec):
        return state[0] - target_x  # Event triggers when x - target_x = 0

    stop_at_target_x.terminal = True
    stop_at_target_x.direction = 1

    # 3. Estimate a safe maximum simulation time
    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)
    
    # 4. Run ODE solver
    sol = solve_ivp(
        model_3d,
        [0, t_max_guess],
        initial_state,
        args=(m, c, g, wind_vec),
        events=stop_at_target_x,
        dense_output=True 
    )

    # 5. Check simulation results and calculate errors
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Success: projectile reached target_x
        final_state = sol.y_events[0][-1]
        
        y_at_target = final_state[1]  # Projectile's actual Y coordinate (lateral deviation)
        z_at_target = final_state[2]  # Projectile's actual Z coordinate (vertical height)
        
        # Targets are y=0 (no deviation), z=target_z
        error_y = y_at_target - 0.0
        error_z = z_at_target - target_z
        
        return [error_y, error_z]
    
    else:
        # Failure: insufficient range
        if sol.y[0].size > 0:
            error_x = sol.y[0][-1] - target_x
        else:
            error_x = -target_x
        return [error_x * 100, error_x * 100]

# --- 3. Function to calculate impact point with given angles ---
def calculate_impact_point(theta_deg, phi_deg, v0, m, c, g, wind_vec):
    """
    Calculate the impact point (x, y, z) when using given launch angles
    Returns the state when projectile hits the ground
    """
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    
    v0z = v0 * np.sin(theta_rad)
    v_proj_xy = v0 * np.cos(theta_rad)
    v0x = v_proj_xy * np.cos(phi_rad)
    v0y = v_proj_xy * np.sin(phi_rad)
    
    initial_state = [0, 0, 0, v0x, v0y, v0z]

    # Define stop event: ground impact (Z=0 or lower)
    def hit_ground(t, state, *args):
        return state[2]  # Track z coordinate (vertical height)
    hit_ground.terminal = True
    hit_ground.direction = -1

    # Estimate maximum time
    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)

    # Run simulator
    sol = solve_ivp(
        model_3d,
        [0, t_max_guess],
        initial_state,
        args=(m, c, g, wind_vec),
        events=hit_ground,
        dense_output=False
    )
    
    # Return the impact state
    if sol.status == 1 and len(sol.y_events[0]) > 0:
        return sol.y_events[0][-1]  # [x, y, z, vx, vy, vz] at impact
    else:
        return None


# --- 4. Main Solver Function ---
def find_launch_angles_3d(rho, wind_vector, target_distance, target_altitude_diff, v0, 
                          m_projectile=5.0, C_D=0.92, projectile_diameter=0.11, g=9.81):
    """
    Calculate the required elevation angle (theta) and azimuth angle (phi) to hit the target
    
    Parameters:
    rho (float): Air density (kg/m^3)
    wind_vector (list/tuple): 3D wind velocity [wx, wy, wz] (m/s) (forward, lateral, vertical)
    target_distance (float): Target horizontal distance (x coordinate) (m)
    target_altitude_diff (float): Target altitude difference (z coordinate) (m)
    v0 (float): Fixed initial velocity (m/s)
    """
    
    # Assign "altitude difference" explicitly to target_z
    target_z = target_altitude_diff
    
    print("--- 3D Ballistic Solver (Z-axis vertical) ---")
    print(f"Target: X={target_distance} m, Y=0 m, Z={target_z} m")
    print(f"Wind (forward, lateral, vertical): [Wx={wind_vector[0]}, Wy={wind_vector[1]}, Wz={wind_vector[2]}] m/s")
    print(f"Air density: {rho} kg/m³")
    
    # 2. Calculate drag constant c
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * rho * A
    
    print(f"Calculation parameters: m={m_projectile} kg, C_D={C_D}, A={A:.4f} m², c={c_drag:.4f}")

    # 3. Prepare solver
    args_tuple = (v0, m_projectile, c_drag, g, wind_vector, 
                  target_distance, target_z)  # Note: passing target_z here
    
    # 4. Provide initial guess [theta_guess, phi_guess]
    initial_guess = [10.0, 0.0] 
    
    print(f"\nSolving... Initial guess: Elevation angle(theta)={initial_guess[0]}°, Azimuth angle(phi)={initial_guess[1]}°")
    
    # 5. Run multivariate root finder
    sol = root(
        objective_function_3d, 
        initial_guess, 
        args=args_tuple, 
        method='lm'
    )
    
    # 6. Return results
    if sol.success:
        theta_sol, phi_sol = sol.x
        print("\n--- ✅ Solution Successful ---")
        print(f"Elevation Angle (Theta, vertical): {theta_sol:.4f} degrees")
        print(f"Azimuth Angle (Phi, horizontal): {phi_sol:.4f} degrees")
        return theta_sol, phi_sol
    else:
        print("\n--- ❌ Solution Failed ---")
        print("Unable to find solution. Please check if target is within range, or try different initial guesses.")
        print(f"Solver message: {sol.message}")
        return None, None

# --- 5. 3D Trajectory Plotting Function ---
def plot_trajectory_3d(theta_deg, phi_deg, v0, m, c, g, wind_vec, target_x, target_z, title_suffix=""):
    """
    Run a complete simulation (until ground impact) using the given launch angles and plot the 3D trajectory.
    Coordinate system: X=forward, Y=lateral, Z=vertical
    """
    print(f"\n--- Generating 3D Trajectory Plot (Theta={theta_deg:.2f}°, Phi={phi_deg:.2f}°) ---")
    
    # 1. Calculate initial velocity
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    
    v0z = v0 * np.sin(theta_rad)
    v_proj_xy = v0 * np.cos(theta_rad)
    v0x = v_proj_xy * np.cos(phi_rad)
    v0y = v_proj_xy * np.sin(phi_rad)
    
    initial_state = [0, 0, 0, v0x, v0y, v0z]

    # 2. Define stop event: ground impact (Z=0 or lower)
    def hit_ground(t, state, *args):
        return state[2]  # Track z coordinate (vertical height)
    hit_ground.terminal = True
    hit_ground.direction = -1

    # 3. Estimate maximum time
    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)

    # 4. Run simulator
    sol = solve_ivp(
        model_3d,
        [0, t_max_guess],
        initial_state,
        args=(m, c, g, wind_vec),
        events=hit_ground,
        dense_output=True
    )
    
    # 5. Generate smooth trajectory data
    if sol.t.size == 0:
        print("Error: Simulation failed to run.")
        return
        
    t_plot = np.linspace(0, sol.t[-1], 300)
    state_plot = sol.sol(t_plot)
    
    x_plot = state_plot[0]  # Forward
    y_plot = state_plot[1]  # Lateral
    z_plot = state_plot[2]  # Vertical

    # 6. Start 3D plotting
    fig = plt.figure(figsize=(12, 10))  # Larger canvas
    ax = fig.add_subplot(111, projection='3d')

    # Plot main trajectory (X=forward, Y=lateral, Z=vertical)
    ax.plot(x_plot, y_plot, z_plot, label='Projectile Trajectory', color='blue', linewidth=2)
    
    # Plot ground shadow (X-Y plane, Z=0)
    ax.plot(x_plot, y_plot, 0, ':', color='grey', label='Ground Projection (Z=0)')

    # Mark launch point
    ax.scatter(0, 0, 0, color='green', s=100, label='Launch Point (0,0,0)', depthshade=True)
    
    # Mark target point (X=target_x, Y=0, Z=target_z)
    ax.scatter(target_x, 0, target_z, color='red', s=100, marker='X', label='Target', depthshade=True)

    # 7. Adjust axis scales
    
    # Dynamically calculate data ranges
    x_max = np.max(x_plot)
    y_abs_max = np.max(np.abs(y_plot))
    z_max = np.max(z_plot)
    z_min = np.min(z_plot)

    # Ensure target point is considered
    x_max = max(x_max, target_x)
    z_max = max(z_max, target_z)
    z_min = min(z_min, target_z)
    
    # Find maximum data range across all dimensions
    range_x = x_max
    range_y = y_abs_max * 2  # Because Y ranges from -Y to +Y
    range_z = z_max - z_min
    
    max_range = max(range_x, range_y, range_z)
    
    # Set X axis (starting from 0)
    ax.set_xlim(0, max_range)
    
    # Set Y axis (centered at 0)
    ax.set_ylim(-max_range / 2, max_range / 2)
    
    # Set Z axis (starting from 0 or lower)
    ax.set_zlim(min(0, z_min), max_range)  # Y-Z range same as X
    
    # Force visual cube
    ax.set_box_aspect([1, 1, 1]) 
    
    # Set axis labels
    ax.set_xlabel('X Axis: Forward Distance (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis: Lateral Deviation (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis: Vertical Height (m)', fontsize=12, labelpad=10)
    
    # Set title
    wind_str = f"Wind(Fwd,Lat,Vert)=[{wind_vec[0]:.0f},{wind_vec[1]:.0f},{wind_vec[2]:.0f}]"
    if title_suffix:
        full_title = f"3D Ballistic Trajectory ({wind_str}) {title_suffix}"
    else:
        full_title = f"3D Ballistic Trajectory ({wind_str})"
    ax.set_title(full_title, fontsize=16)
    
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.6)  # Add grid lines
    plt.show()


# --- 6. Wind Sensitivity Analysis Table ---
def generate_wind_sensitivity_table(target_x, target_z, v0, m_projectile, c_drag, g, 
                                   wx_range=range(-10, 11), wy_range=range(0, 11), air_density=1.17):
    """
    Generate a table showing the impact of different wind conditions on targeting accuracy
    """
    print("\n" + "="*60)
    print("WIND SENSITIVITY ANALYSIS")
    print("="*60)
    
    # First, calculate no-wind solution as reference
    print("Calculating no-wind reference solution...")
    wind_none = [0.0, 0.0, 0.0]
    theta_ref, phi_ref = find_launch_angles_3d(
        rho=air_density,
        wind_vector=wind_none,
        target_distance=target_x,
        target_altitude_diff=target_z,
        v0=v0,
        m_projectile=m_projectile,
        C_D=0.92,
        projectile_diameter=0.11,
        g=g
    )
    
    if theta_ref is None:
        print("Failed to calculate reference solution. Cannot proceed with sensitivity analysis.")
        return
    
    print(f"Reference angles (no wind): Theta={theta_ref:.4f}°, Phi={phi_ref:.4f}°")
    
    # Initialize results table
    results = []
    
    print("\nCalculating wind effects...")
    # Test Wx variations (Wy=0)
    for wx in tqdm(wx_range, desc="Wx variations"):
        wind_vec = [wx, 0.0, 0.0]
        impact_state = calculate_impact_point(theta_ref, phi_ref, v0, m_projectile, c_drag, g, wind_vec)
        
        if impact_state is not None:
            impact_x, impact_y, impact_z = impact_state[0], impact_state[1], impact_state[2]
            x_error = impact_x - target_x
            y_error = impact_y - 0.0
            z_error = impact_z - target_z
            horizontal_error = np.sqrt(x_error**2 + y_error**2)
            
            results.append({
                'Wind Type': 'Wx',
                'Wind Speed (m/s)': wx,
                'X Error (m)': x_error,
                'Y Error (m)': y_error,
                'Z Error (m)': z_error,
                'Horizontal Error (m)': horizontal_error
            })
    
    # Test Wy variations (Wx=0)
    for wy in tqdm(wy_range, desc="Wy variations"):
        wind_vec = [0.0, wy, 0.0]
        impact_state = calculate_impact_point(theta_ref, phi_ref, v0, m_projectile, c_drag, g, wind_vec)
        
        if impact_state is not None:
            impact_x, impact_y, impact_z = impact_state[0], impact_state[1], impact_state[2]
            x_error = impact_x - target_x
            y_error = impact_y - 0.0
            z_error = impact_z - target_z
            horizontal_error = np.sqrt(x_error**2 + y_error**2)
            
            results.append({
                'Wind Type': 'Wy',
                'Wind Speed (m/s)': wy,
                'X Error (m)': x_error,
                'Y Error (m)': y_error,
                'Z Error (m)': z_error,
                'Horizontal Error (m)': horizontal_error
            })
    
    # Display results table
    print("\n" + "="*80)
    print("WIND SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    print(f"Target: X={target_x}m, Z={target_z}m | Reference angles: Theta={theta_ref:.2f}°, Phi={phi_ref:.2f}°")
    print("\nWind Effects Table:")
    print("-" * 100)
    print(f"{'Wind Type':<10} {'Speed (m/s)':<12} {'X Error (m)':<12} {'Y Error (m)':<12} {'Z Error (m)':<12} {'Horiz Error (m)':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['Wind Type']:<10} {result['Wind Speed (m/s)']:<12} {result['X Error (m)']:<12.2f} "
              f"{result['Y Error (m)']:<12.2f} {result['Z Error (m)']:<12.2f} {result['Horizontal Error (m)']:<15.2f}")
    
    # Create visualizations
    # Separate Wx and Wy results
    wx_results = [r for r in results if r['Wind Type'] == 'Wx']
    wy_results = [r for r in results if r['Wind Type'] == 'Wy']
    
    # Plot 1: X and Y errors vs wind speed
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    wx_speeds = [r['Wind Speed (m/s)'] for r in wx_results]
    wx_x_errors = [r['X Error (m)'] for r in wx_results]
    wy_speeds = [r['Wind Speed (m/s)'] for r in wy_results]
    wy_y_errors = [r['Y Error (m)'] for r in wy_results]
    
    plt.plot(wx_speeds, wx_x_errors, 'ro-', linewidth=2, markersize=6, label='Wx: X Error (Range)')
    plt.plot(wy_speeds, wy_y_errors, 'bs-', linewidth=2, markersize=6, label='Wy: Y Error (Lateral)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Error (m)')
    plt.title('Range and Lateral Errors vs Wind Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Horizontal error vs wind speed
    plt.subplot(1, 3, 2)
    wx_horiz_errors = [r['Horizontal Error (m)'] for r in wx_results]
    wy_horiz_errors = [r['Horizontal Error (m)'] for r in wy_results]
    
    plt.plot(wx_speeds, wx_horiz_errors, 'ro-', linewidth=2, markersize=6, label='Wx (Head/Tail Wind)')
    plt.plot(wy_speeds, wy_horiz_errors, 'bs-', linewidth=2, markersize=6, label='Wy (Cross Wind)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Horizontal Error (m)')
    plt.title('Total Horizontal Error vs Wind Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Error comparison as percentage of target distance
    plt.subplot(1, 3, 3)
    wx_error_pct = [100 * r['Horizontal Error (m)'] / target_x for r in wx_results]
    wy_error_pct = [100 * r['Horizontal Error (m)'] / target_x for r in wy_results]
    
    plt.plot(wx_speeds, wx_error_pct, 'ro-', linewidth=2, markersize=6, label='Wx (Head/Tail Wind)')
    plt.plot(wy_speeds, wy_error_pct, 'bs-', linewidth=2, markersize=6, label='Wy (Cross Wind)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Error (% of Target Distance)')
    plt.title('Horizontal Error as Percentage of Target Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    if wx_results:
        max_wx_error = max([abs(r['Horizontal Error (m)']) for r in wx_results])
        avg_wx_error = np.mean([abs(r['Horizontal Error (m)']) for r in wx_results])
        print(f"Wx (Head/Tail Wind):")
        print(f"  Max horizontal error: {max_wx_error:.2f} m ({100*max_wx_error/target_x:.1f}% of target)")
        print(f"  Average horizontal error: {avg_wx_error:.2f} m")
        
        # Find headwind and tailwind effects
        headwind_errors = [r for r in wx_results if r['Wind Speed (m/s)'] < 0]
        tailwind_errors = [r for r in wx_results if r['Wind Speed (m/s)'] > 0]
        
        if headwind_errors:
            avg_headwind_error = np.mean([abs(r['Horizontal Error (m)']) for r in headwind_errors])
            print(f"  Average headwind error: {avg_headwind_error:.2f} m")
        
        if tailwind_errors:
            avg_tailwind_error = np.mean([abs(r['Horizontal Error (m)']) for r in tailwind_errors])
            print(f"  Average tailwind error: {avg_tailwind_error:.2f} m")
    
    if wy_results:
        max_wy_error = max([r['Horizontal Error (m)'] for r in wy_results])
        avg_wy_error = np.mean([r['Horizontal Error (m)'] for r in wy_results])
        print(f"Wy (Cross Wind):")
        print(f"  Max horizontal error: {max_wy_error:.2f} m ({100*max_wy_error/target_x:.1f}% of target)")
        print(f"  Average horizontal error: {avg_wy_error:.2f} m")
    
    return results, wx_results, wy_results


# --- 7. Main Program ---
if __name__ == "__main__":
    
    # --- 1. Input your parameters here ---
    
    # Physical environment
    air_density = 1.17                 # Air density (kg/m³)
    
    # Wind velocity [Wx(forward/backward), Wy(left/right), Wz(up/down)] m/s
    wind = [0.0, 10.0, 0.0]  # 10m/s crosswind (from left to right)
    
    # Target 
    target_x = 1200.0                   # Distance (m)
    target_z = 0.0                      # Altitude difference (m) (Z axis)
    
    # Projectile 
    v_initial = 450.0                   # Initial velocity (m/s)

    # --- 2. Other parameters 
    m_projectile = 5.0      # Mass (kg)
    C_D = 0.92      # Drag coefficient
    projectile_diameter = 0.11  # Diameter (m)
    g = 9.81        # (m/s^2)
    
    # 3. Calculate drag constant c
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * air_density * A
    
    # ----------------------------------
    
    print("="*60)
    print(f"Simplified 3D Analysis (v0={v_initial}, x={target_x}, z={target_z})")
    print("="*60)
    
    # 4. Run solver (with wind)
    theta, phi = find_launch_angles_3d(
        rho=air_density,
        wind_vector=wind,
        target_distance=target_x,
        target_altitude_diff=target_z,
        v0=v_initial,
        m_projectile=m_projectile,
        C_D=C_D,
        projectile_diameter=projectile_diameter,
        g=g
    )
    
    # --- 5. If successful, plot 3D trajectory (with wind) ---
    if theta is not None:
        plot_trajectory_3d(
            theta, phi, v_initial,
            m_projectile, c_drag, g,
            wind,
            target_x, target_z,  # <--- Pass target_z
            title_suffix="(With Wind)"
        )
    
    # 6. Run no-wind case for comparison
    if theta is not None:
        print("\n" + "="*40)
        print("For comparison: Calculating no-wind case...")
        wind_none = [0.0, 0.0, 0.0]
        theta_0, phi_0 = find_launch_angles_3d(
            rho=air_density,
            wind_vector=wind_none,
            target_distance=target_x,
            target_altitude_diff=target_z,
            v0=v_initial,
            m_projectile=m_projectile,
            C_D=C_D,
            projectile_diameter=projectile_diameter,
            g=g
        )
        if theta_0 is not None:
            print("\n--- Final Result Comparison ---")
            print(f"With Wind (Wy={wind[1]}): Elevation angle={theta:.4f}°, Azimuth angle={phi:.4f}°")
            print(f"No Wind (Wy=0): Elevation angle={theta_0:.4f}°, Azimuth angle={phi_0:.4f}°")
            
            print("\nAnalysis:")
            print(f" * Conclusion 1 (Phi): Crosswind forces you to aim into the wind by {phi:.4f} degrees to counteract drift.")
            print(f" * Conclusion 2 (Theta): Crosswind also increases total drag, forcing you to raise the elevation angle by "
                  f"{(theta - theta_0):.4f} degrees.")
                  
            # --- 7. Plot 3D trajectory for no-wind case ---
            plot_trajectory_3d(
                theta_0, phi_0, v_initial,
                m_projectile, c_drag, g,
                wind_none,
                target_x, target_z,  # <--- Pass target_z
                title_suffix="(No Wind Comparison)"
            )
            
            # --- 8. Calculate error if using no-wind angles in windy conditions ---
            print("\n" + "="*50)
            print("ERROR ANALYSIS: Using No-Wind Angles in Windy Conditions")
            print("="*50)
            
            # Calculate impact point when using no-wind angles in windy conditions
            impact_state = calculate_impact_point(theta_0, phi_0, v_initial, m_projectile, c_drag, g, wind)
            
            if impact_state is not None:
                impact_x, impact_y, impact_z = impact_state[0], impact_state[1], impact_state[2]
                
                print(f"\nIf you use no-wind angles (Theta={theta_0:.4f}°, Phi={phi_0:.4f}°) in windy conditions:")
                print(f"Impact point: X={impact_x:.2f} m, Y={impact_y:.2f} m, Z={impact_z:.2f} m")
                print(f"Target point: X={target_x:.2f} m, Y=0 m, Z={target_z:.2f} m")
                
                # Calculate errors
                x_error = impact_x - target_x
                y_error = impact_y - 0.0
                z_error = impact_z - target_z
                
                print(f"\nErrors:")
                print(f"  X-error (range): {x_error:.2f} m ({x_error/10:.1f}% of target distance)")
                print(f"  Y-error (lateral): {y_error:.2f} m")
                print(f"  Z-error (vertical): {z_error:.2f} m")
                
                # Calculate miss distance (2D horizontal miss)
                horizontal_miss = np.sqrt(x_error**2 + y_error**2)
                print(f"  Horizontal miss distance: {horizontal_miss:.2f} m")
                
                # Calculate total miss distance (3D)
                total_miss = np.sqrt(x_error**2 + y_error**2 + z_error**2)
                print(f"  Total miss distance (3D): {total_miss:.2f} m")
                
                # Additional analysis
                print(f"\nAdditional Analysis:")
                print(f"  Lateral drift due to wind: {y_error:.2f} m ({y_error/10:.1f}% of target distance)")
                if x_error > 0:
                    print(f"  Projectile overshoots target by {x_error:.2f} m")
                else:
                    print(f"  Projectile falls short by {-x_error:.2f} m")
                    
                # Plot this error case
                plot_trajectory_3d(
                    theta_0, phi_0, v_initial,
                    m_projectile, c_drag, g,
                    wind,
                    target_x, target_z,
                    title_suffix="(No-Wind Angles in Windy Conditions - ERROR)"
                )
    
    # --- 9. Wind Sensitivity Analysis ---
    print("\n" + "="*60)
    print("STARTING WIND SENSITIVITY ANALYSIS")
    print("="*60)

    # 调用风敏感性分析函数，Wx范围从-10到10
    wind_results, wx_results, wy_results = generate_wind_sensitivity_table(
        target_x=target_x,
        target_z=target_z,
        v0=v_initial,
        m_projectile=m_projectile,
        c_drag=c_drag,
        g=g,
        wx_range=range(-10, 11),  # Wx从-10到10
        wy_range=range(0, 11)     # Wy从0到10
    )