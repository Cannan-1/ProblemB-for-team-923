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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
                          m_projectile=5.0, C_D=0.92, projectile_diameter=0.11, g=9.81,
                          verbose=True): # *** NEW verbose parameter ***
    """
    Calculate the required elevation angle (theta) and azimuth angle (phi) to hit the target
    
    Parameters:
    rho (float): Air density (kg/m^3)
    wind_vector (list/tuple): 3D wind velocity [wx, wy, wz] (m/s) (forward, lateral, vertical)
    target_distance (float): Target horizontal distance (x coordinate) (m)
    target_altitude_diff (float): Target altitude difference (z coordinate) (m)
    v0 (float): Fixed initial velocity (m/s)
    verbose (bool): If True, print solving progress and results
    """
    
    # Assign "altitude difference" explicitly to target_z
    target_z = target_altitude_diff
    
    if verbose:
        print("--- 3D Ballistic Solver (Z-axis vertical) ---")
        print(f"Target: X={target_distance} m, Y=0 m, Z={target_z} m")
        print(f"Wind (forward, lateral, vertical): [Wx={wind_vector[0]}, Wy={wind_vector[1]}, Wz={wind_vector[2]}] m/s")
        print(f"Air density: {rho} kg/m³")
    
    # 2. Calculate drag constant c
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * rho * A
    
    if verbose:
        print(f"Calculation parameters: m={m_projectile} kg, C_D={C_D}, A={A:.4f} m², c={c_drag:.4f}")

    # 3. Prepare solver
    args_tuple = (v0, m_projectile, c_drag, g, wind_vector, 
                  target_distance, target_z)  # Note: passing target_z here
    
    # 4. Provide initial guess [theta_guess, phi_guess]
    initial_guess = [10.0, 0.0] 
    
    if verbose:
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
        if verbose:
            print("\n--- ✅ Solution Successful ---")
            print(f"Elevation Angle (Theta, vertical): {theta_sol:.4f} degrees")
            print(f"Azimuth Angle (Phi, horizontal): {phi_sol:.4f} degrees")
        return theta_sol, phi_sol
    else:
        if verbose:
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
                                    wx_range=range(-10, 11), wy_range=range(0, 11), air_density=1.17,
                                    height_label=""):
    """
    Generate a table showing the impact of different wind conditions on targeting accuracy
    """
    print("\n" + "="*60)
    print(f"WIND SENSITIVITY ANALYSIS - Height: {target_z}m {height_label}")
    print("="*60)
    
    # First, calculate no-wind solution as reference
    print("Calculating no-wind reference solution...")
    wind_none = [0.0, 0.0, 0.0]
    
    # Calculate C_D and projectile_diameter from c_drag for the find_launch_angles_3d function
    # c_drag = 0.5 * C_D * rho * A 
    # A = np.pi * (projectile_diameter / 2)**2
    # This is a bit backward, let's just pass the C_D and diameter
    # Assuming standard parameters as the function doesn't get them
    C_D_assumed = 0.92
    projectile_diameter_assumed = 0.11
    
    theta_ref, phi_ref = find_launch_angles_3d(
        rho=air_density,
        wind_vector=wind_none,
        target_distance=target_x,
        target_altitude_diff=target_z,
        v0=v0,
        m_projectile=m_projectile,
        C_D=C_D_assumed, # Pass C_D
        projectile_diameter=projectile_diameter_assumed, # Pass diameter
        g=g,
        verbose=True # Be verbose for this reference calculation
    )
    
    if theta_ref is None:
        print("Failed to calculate reference solution. Cannot proceed with sensitivity analysis.")
        return None, None, None, None, None
    
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
    print(f"WIND SENSITIVITY ANALYSIS RESULTS - Height: {target_z}m {height_label}")
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
    plt.title(f'Range and Lateral Errors vs Wind Speed\n(Height: {target_z}m {height_label})')
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
    plt.title(f'Total Horizontal Error vs Wind Speed\n(Height: {target_z}m {height_label})')
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
    plt.title(f'Horizontal Error as Percentage of Target Distance\n(Height: {target_z}m {height_label})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print(f"SUMMARY STATISTICS - Height: {target_z}m {height_label}")
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
    
    return results, wx_results, wy_results, theta_ref, phi_ref


# --- 7. Linear Regression Analysis ---
def perform_linear_regression_analysis(all_results, target_x):
    """
    Perform linear regression analysis on wind error data
    Extract linear relationships between wind speed and error
    """
    print("\n" + "="*80)
    print("LINEAR REGRESSION ANALYSIS: Wind Speed vs Error")
    print("="*80)
    
    regression_results = {}
    
    # Colors for different heights
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    heights = list(all_results.keys())
    
    # Prepare data for regression
    for height in heights:
        if height in all_results and all_results[height]['wx_results'] and all_results[height]['wy_results']:
            print(f"\n--- Height: {height}m ---")
            
            # Wx regression (head/tail wind vs range error)
            wx_speeds = np.array([r['Wind Speed (m/s)'] for r in all_results[height]['wx_results']]).reshape(-1, 1)
            wx_x_errors = np.array([r['X Error (m)'] for r in all_results[height]['wx_results']])
            
            # Wy regression (cross wind vs lateral error)
            wy_speeds = np.array([r['Wind Speed (m/s)'] for r in all_results[height]['wy_results']]).reshape(-1, 1)
            wy_y_errors = np.array([r['Y Error (m)'] for r in all_results[height]['wy_results']])
            
            # Perform linear regression
            wx_model = LinearRegression()
            wx_model.fit(wx_speeds, wx_x_errors)
            wx_r2 = r2_score(wx_x_errors, wx_model.predict(wx_speeds))
            
            wy_model = LinearRegression()
            wy_model.fit(wy_speeds, wy_y_errors)
            wy_r2 = r2_score(wy_y_errors, wy_model.predict(wy_speeds))
            
            # Store results
            regression_results[height] = {
                'wx_slope': wx_model.coef_[0],
                'wx_intercept': wx_model.intercept_,
                'wx_r2': wx_r2,
                'wy_slope': wy_model.coef_[0],
                'wy_intercept': wy_model.intercept_,
                'wy_r2': wy_r2
            }
            
            # Print results
            print(f"Wx (Head/Tail Wind):")
            print(f"  Slope: {wx_model.coef_[0]:.4f} m/(m/s) (Range error per m/s wind)")
            print(f"  Intercept: {wx_model.intercept_:.4f} m")
            print(f"  R²: {wx_r2:.4f}")
            
            print(f"Wy (Cross Wind):")
            print(f"  Slope: {wy_model.coef_[0]:.4f} m/(m/s) (Lateral error per m/s wind)")
            print(f"  Intercept: {wy_model.intercept_:.4f} m")
            print(f"  R²: {wy_r2:.4f}")
            
            # Calculate error per 1 m/s wind
            print(f"Summary: For every 1 m/s of wind:")
            print(f"  Head/Tail wind (Wx) causes {abs(wx_model.coef_[0]):.2f} m range error")
            print(f"  Cross wind (Wy) causes {abs(wy_model.coef_[0]):.2f} m lateral error")
    
    # Create regression visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Wx regression lines
    plt.subplot(2, 3, 1)
    for i, height in enumerate(heights):
        if height in regression_results:
            # Generate points for regression line
            wx_range = np.array([-10, 10]).reshape(-1, 1)
            wx_pred = regression_results[height]['wx_slope'] * wx_range + regression_results[height]['wx_intercept']
            
            plt.plot(wx_range, wx_pred, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m (slope={regression_results[height]["wx_slope"]:.2f})')
    
    plt.xlabel('Wx Wind Speed (m/s)')
    plt.ylabel('X Error (Range Error, m)')
    plt.title('Linear Regression: Range Error vs Head/Tail Wind')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Wy regression lines
    plt.subplot(2, 3, 2)
    for i, height in enumerate(heights):
        if height in regression_results:
            # Generate points for regression line
            wy_range = np.array([0, 10]).reshape(-1, 1)
            wy_pred = regression_results[height]['wy_slope'] * wy_range + regression_results[height]['wy_intercept']
            
            plt.plot(wy_range, wy_pred, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m (slope={regression_results[height]["wy_slope"]:.2f})')
    
    plt.xlabel('Wy Wind Speed (m/s)')
    plt.ylabel('Y Error (Lateral Error, m)')
    plt.title('Linear Regression: Lateral Error vs Cross Wind')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: R² values comparison
    plt.subplot(2, 3, 3)
    wx_r2_values = [regression_results[h]['wx_r2'] for h in heights if h in regression_results]
    wy_r2_values = [regression_results[h]['wy_r2'] for h in heights if h in regression_results]
    
    x_pos = np.arange(len(heights))
    width = 0.35
    
    plt.bar(x_pos - width/2, wx_r2_values, width, label='Wx R²', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, wy_r2_values, width, label='Wy R²', color='blue', alpha=0.7)
    
    plt.xlabel('Target Height (m)')
    plt.ylabel('R² Value')
    plt.title('Regression Goodness-of-Fit (R²) by Height')
    plt.xticks(x_pos, heights)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Slope comparison
    plt.subplot(2, 3, 4)
    wx_slopes = [abs(regression_results[h]['wx_slope']) for h in heights if h in regression_results]
    wy_slopes = [abs(regression_results[h]['wy_slope']) for h in heights if h in regression_results]
    
    plt.bar(x_pos - width/2, wx_slopes, width, label='Wx Slope (m/(m/s))', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, wy_slopes, width, label='Wy Slope (m/(m/s))', color='blue', alpha=0.7)
    
    plt.xlabel('Target Height (m)')
    plt.ylabel('Slope (m/(m/s))')
    plt.title('Regression Slopes by Height\n(Error per m/s wind)')
    plt.xticks(x_pos, heights)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Error per 1 m/s wind as percentage of target
    plt.subplot(2, 3, 5)
    wx_error_pct = [100 * abs(regression_results[h]['wx_slope']) / target_x for h in heights if h in regression_results]
    wy_error_pct = [100 * abs(regression_results[h]['wy_slope']) / target_x for h in heights if h in regression_results]
    
    plt.bar(x_pos - width/2, wx_error_pct, width, label='Wx Error (%/m/s)', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, wy_error_pct, width, label='Wy Error (%/m/s)', color='blue', alpha=0.7)
    
    plt.xlabel('Target Height (m)')
    plt.ylabel('Error (% of target per m/s wind)')
    plt.title('Error Sensitivity as Percentage of Target\n(Per 1 m/s wind)')
    plt.xticks(x_pos, heights)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Height vs slope relationship
    plt.subplot(2, 3, 6)
    plt.plot(heights, wx_slopes, 'ro-', linewidth=2, markersize=6, label='Wx Slope')
    plt.plot(heights, wy_slopes, 'bs-', linewidth=2, markersize=6, label='Wy Slope')
    plt.xlabel('Target Height (m)')
    plt.ylabel('Slope (m/(m/s))')
    plt.title('Wind Sensitivity vs Target Height')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive regression summary
    print("\n" + "="*80)
    print("COMPREHENSIVE REGRESSION SUMMARY")
    print("="*80)
    
    for height in heights:
        if height in regression_results:
            print(f"\nHeight: {height}m")
            print(f"  Wx (Head/Tail Wind):")
            print(f"    Error = {regression_results[height]['wx_slope']:.4f} × WindSpeed + {regression_results[height]['wx_intercept']:.4f}")
            print(f"    R² = {regression_results[height]['wx_r2']:.4f}")
            print(f"    1 m/s wind → {abs(regression_results[height]['wx_slope']):.2f} m range error")
            
            print(f"  Wy (Cross Wind):")
            print(f"    Error = {regression_results[height]['wy_slope']:.4f} × WindSpeed + {regression_results[height]['wy_intercept']:.4f}")
            print(f"    R² = {regression_results[height]['wy_r2']:.4f}")
            print(f"    1 m/s wind → {abs(regression_results[height]['wy_slope']):.2f} m lateral error")
    
    return regression_results


# --- 8. Multi-Height Analysis Function ---
def multi_height_wind_analysis(target_x, v0, m_projectile, c_drag, g, 
                               heights=[-100, -50, 0, 50, 100], 
                               wx_range=range(-10, 11), wy_range=range(0, 11), air_density=1.17):
    """
    Perform wind sensitivity analysis for multiple target heights
    """
    print("\n" + "="*80)
    print("MULTI-HEIGHT WIND SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Store results for all heights
    all_results = {}
    
    # Colors for different heights
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, height in enumerate(heights):
        print(f"\n\n{'='*60}")
        print(f"ANALYZING HEIGHT: {height}m")
        print(f"{'='*60}")
        
        # Determine height label for display
        if height < 0:
            height_label = "(Below Launch)"
        elif height == 0:
            height_label = "(Same Level)"
        else:
            height_label = "(Above Launch)"
        
        # Perform wind sensitivity analysis for this height
        results, wx_results, wy_results, theta_ref, phi_ref = generate_wind_sensitivity_table(
            target_x=target_x,
            target_z=height,
            v0=v0,
            m_projectile=m_projectile,
            c_drag=c_drag,
            g=g,
            wx_range=wx_range,
            wy_range=wy_range,
            air_density=air_density,
            height_label=height_label
        )
        
        # Store results
        all_results[height] = {
            'results': results,
            'wx_results': wx_results,
            'wy_results': wy_results,
            'theta_ref': theta_ref,
            'phi_ref': phi_ref
        }
    
    # Create comparison plots across all heights
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS ACROSS ALL HEIGHTS")
    print("="*80)
    
    # Plot 1: Wx errors comparison across heights
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for i, height in enumerate(heights):
        if height in all_results and all_results[height]['wx_results']:
            wx_speeds = [r['Wind Speed (m/s)'] for r in all_results[height]['wx_results']]
            wx_errors = [r['X Error (m)'] for r in all_results[height]['wx_results']]
            plt.plot(wx_speeds, wx_errors, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m')
    
    plt.xlabel('Wx Wind Speed (m/s)')
    plt.ylabel('X Error (Range Error, m)')
    plt.title('Range Error vs Head/Tail Wind Speed\n(Comparison Across Heights)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Wy errors comparison across heights
    plt.subplot(1, 3, 2)
    for i, height in enumerate(heights):
        if height in all_results and all_results[height]['wy_results']:
            wy_speeds = [r['Wind Speed (m/s)'] for r in all_results[height]['wy_results']]
            wy_errors = [r['Y Error (m)'] for r in all_results[height]['wy_results']]
            plt.plot(wy_speeds, wy_errors, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m')
    
    plt.xlabel('Wy Wind Speed (m/s)')
    plt.ylabel('Y Error (Lateral Error, m)')
    plt.title('Lateral Error vs Cross Wind Speed\n(Comparison Across Heights)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Maximum horizontal errors by height
    plt.subplot(1, 3, 3)
    max_wx_errors = []
    max_wy_errors = []
    
    for height in heights:
        if height in all_results:
            if all_results[height]['wx_results']:
                max_wx = max([abs(r['Horizontal Error (m)']) for r in all_results[height]['wx_results']])
            else:
                max_wx = 0
                
            if all_results[height]['wy_results']:
                # *** THIS WAS THE ERROR (FIXED) ***
                max_wy = max([r['Horizontal Error (m)'] for r in all_results[height]['wy_results']])
            else:
                max_wy = 0
                
            max_wx_errors.append(max_wx)
            max_wy_errors.append(max_wy)
    
    x_pos = np.arange(len(heights))
    width = 0.35
    
    plt.bar(x_pos - width/2, max_wx_errors, width, label='Max Wx Error', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, max_wy_errors, width, label='Max Wy Error', color='blue', alpha=0.7)
    
    plt.xlabel('Target Height (m)')
    plt.ylabel('Maximum Horizontal Error (m)')
    plt.title('Maximum Horizontal Errors by Target Height')
    plt.xticks(x_pos, heights)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY ACROSS ALL HEIGHTS")
    print("="*80)
    
    for height in heights:
        if height in all_results:
            print(f"\nHeight: {height}m")
            if all_results[height]['wx_results']:
                max_wx_error = max([abs(r['Horizontal Error (m)']) for r in all_results[height]['wx_results']])
                avg_wx_error = np.mean([abs(r['Horizontal Error (m)']) for r in all_results[height]['wx_results']])
                print(f"  Wx (Head/Tail Wind): Max={max_wx_error:.2f}m, Avg={avg_wx_error:.2f}m")
            
            if all_results[height]['wy_results']:
                # *** THESE TWO LINES WERE THE ERROR (FIXED) ***
                max_wy_error = max([r['Horizontal Error (m)'] for r in all_results[height]['wy_results']])
                avg_wy_error = np.mean([r['Horizontal Error (m)'] for r in all_results[height]['wy_results']])
                print(f"  Wy (Cross Wind):     Max={max_wy_error:.2f}m, Avg={avg_wy_error:.2f}m")
            
            print(f"  Reference angles: Theta={all_results[height]['theta_ref']:.2f}°, Phi={all_results[height]['phi_ref']:.2f}°")
    
    return all_results

# --- 9. Azimuth Correction Analysis ---
# *** NEW FUNCTION (UNCHANGED) ***
def calculate_azimuth_corrections(all_heights_results, v0, m_projectile, C_D, projectile_diameter, g, air_density, target_x):
    """
    Calculate the required azimuth (phi) correction for Wy=5 and Wy=10 m/s at each height.
    """
    print("\n" + "="*103)
    print("AZIMUTH (PHI) CORRECTION ANALYSIS FOR CROSSWIND (Wy)")
    print("="*103)
    print(f"{'Height (m)':<12} {'Phi (No Wind)':<15} {'Phi (Wy=5 m/s)':<18} {'Correction (Wy=5)':<20} {'Phi (Wy=10 m/s)':<18} {'Correction (Wy=10)':<20}")
    print("-" * 103)
    
    correction_data = {}
    
    for height, data in all_heights_results.items():
        phi_ref = data['phi_ref']
        
        if phi_ref is None:
            print(f"{height:<12} {'N/A (No Ref)':<15}")
            continue
            
        target_z = height
        
        # --- Calculate for Wy = 5 m/s ---
        wind_y5 = [0.0, 5.0, 0.0]
        theta_y5, phi_y5 = find_launch_angles_3d(
            air_density, wind_y5, target_x, target_z, v0, 
            m_projectile, C_D, projectile_diameter, g,
            verbose=False # Set verbose=False to keep output clean
        )
        
        # --- Calculate for Wy = 10 m/s ---
        wind_y10 = [0.0, 10.0, 0.0]
        theta_y10, phi_y10 = find_launch_angles_3d(
            air_density, wind_y10, target_x, target_z, v0, 
            m_projectile, C_D, projectile_diameter, g,
            verbose=False # Set verbose=False
        )
        
        # --- Calculate corrections ---
        if phi_y5 is not None:
            correction_y5 = phi_y5 - phi_ref
        else:
            phi_y5 = np.nan # Use nan for failed calculations
            correction_y5 = np.nan

        if phi_y10 is not None:
            correction_y10 = phi_y10 - phi_ref
        else:
            phi_y10 = np.nan
            correction_y10 = np.nan
            
        # Store data
        correction_data[height] = {
            'phi_ref': phi_ref,
            'phi_y5': phi_y5,
            'correction_y5': correction_y5,
            'phi_y10': phi_y10,
            'correction_y10': correction_y10
        }
        
        # Print results in the table
        print(f"{height:<12} {phi_ref:<15.4f} {phi_y5:<18.4f} {correction_y5:<20.4f} {phi_y10:<18.4f} {correction_y10:<20.4f}")
        
    print("-" * 103)
    return correction_data


# --- 10. Main Program ---
if __name__ == "__main__":
    
    # --- 1. Input your parameters here ---
    
    # Physical environment
    air_density = 1.17           # Air density (kg/m³)
    
    # Target 
    target_x = 1200.0            # Distance (m)
    
    # Projectile 
    v_initial = 450.0            # Initial velocity (m/s)

    # --- 2. Other parameters 
    m_projectile = 5.0    # Mass (kg)
    C_D = 0.92    # Drag coefficient
    projectile_diameter = 0.11   # Diameter (m)
    g = 9.81        # (m/s^2)
    
    # 3. Calculate drag constant c
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * air_density * A
    
    # ----------------------------------
    
    print("="*60)
    print(f"MULTI-HEIGHT WIND SENSITIVITY ANALYSIS")
    print(f"Target Distance: {target_x}m, Initial Velocity: {v_initial}m/s")
    print("="*60)
    
    # --- 10. Multi-Height Wind Sensitivity Analysis ---
    all_heights_results = multi_height_wind_analysis(
        target_x=target_x,
        v0=v_initial,
        m_projectile=m_projectile,
        c_drag=c_drag,
        g=g,
        heights=[-100, -50, 0, 50, 100],
        wx_range=range(-10, 11),
        wy_range=range(0, 11),
        air_density=air_density
    )
    
    # --- 11. Linear Regression Analysis ---
    regression_results = perform_linear_regression_analysis(all_heights_results, target_x)
    
    # --- 12. Azimuth Correction Analysis ---
    # *** NEWLY ADDED CALL ***
    calculate_azimuth_corrections(
        all_heights_results,
        v_initial,
        m_projectile,
        C_D,
        projectile_diameter,
        g,
        air_density,
        target_x
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Summary of key findings:")
    print("1. Higher target heights generally require higher elevation angles")
    print("2. Cross wind (Wy) effects are relatively consistent across heights")
    print("3. Head/tail wind (Wx) effects may vary with target height")
    print("4. Negative heights (below launch) may have different sensitivity patterns")
    print("5. Linear regression provides quantitative wind sensitivity coefficients")
    print("6. Azimuth correction table provides specific angular adjustments for crosswind")