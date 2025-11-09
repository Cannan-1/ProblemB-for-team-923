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


# --- 1. 3D Physical Model (Unchanged) ---
def model_3d(t, state, m, c, g, wind_vec):
    vx, vy, vz = state[3], state[4], state[5]
    wx, wy, wz = wind_vec
    v_rel_x = vx - wx
    v_rel_y = vy - wy
    v_rel_z = vz - wz
    v_rel = (v_rel_x**2 + v_rel_y**2 + v_rel_z**2)**0.5
    
    if v_rel < 1e-9:
        ax, ay, az = 0, 0, -g
    else:
        ax = -(c / m) * v_rel * v_rel_x
        ay = -(c / m) * v_rel * v_rel_y
        az = -g - (c / m) * v_rel * v_rel_z
    return [vx, vy, vz, ax, ay, az]

# --- 2. Objective Function (Unchanged) ---
# (Used by the solver 'find_launch_angles_3d')
def objective_function_3d(launch_angles, v0, m, c, g, wind_vec, target_x, target_z):
    theta_deg, phi_deg = launch_angles
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    
    v0z = v0 * np.sin(theta_rad)
    v_proj_xy = v0 * np.cos(theta_rad)
    v0x = v_proj_xy * np.cos(phi_rad)
    v0y = v_proj_xy * np.sin(phi_rad)
    initial_state = [0, 0, 0, v0x, v0y, v0z]
    
    def stop_at_target_x(t, state, m, c, g, wind_vec):
        return state[0] - target_x
    stop_at_target_x.terminal = True
    stop_at_target_x.direction = 1

    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)
    
    sol = solve_ivp(
        model_3d, [0, t_max_guess], initial_state,
        args=(m, c, g, wind_vec),
        events=stop_at_target_x,
        dense_output=True 
    )

    if sol.status == 1 and len(sol.t_events[0]) > 0:
        final_state = sol.y_events[0][-1]
        y_at_target = final_state[1]
        z_at_target = final_state[2]
        error_y = y_at_target - 0.0
        error_z = z_at_target - target_z
        return [error_y, error_z]
    else:
        error_x = sol.y[0][-1] - target_x if sol.y[0].size > 0 else -target_x
        return [error_x * 100, error_x * 100]

# --- 3. Calculate Impact Point (MODIFIED) ---
def calculate_impact_point(theta_deg, phi_deg, v0, m, c, g, wind_vec, target_z):
    """
    Calculate the impact state (x, y, z) when using given launch angles
    *** MODIFIED ***
    Returns the state when projectile hits the TARGET'S altitude (z=target_z)
    """
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    
    v0z = v0 * np.sin(theta_rad)
    v_proj_xy = v0 * np.cos(theta_rad)
    v0x = v_proj_xy * np.cos(phi_rad)
    v0y = v_proj_xy * np.sin(phi_rad)
    initial_state = [0, 0, 0, v0x, v0y, v0z]

    # *** MODIFIED EVENT ***
    # Stop when the projectile's height (state[2]) crosses the target's height (target_z)
    def hit_target_altitude(t, state, *args):
        return state[2] - target_z  # Stop when z = target_z
    
    hit_target_altitude.terminal = True
    hit_target_altitude.direction = -1 # Stop when z is decreasing (i.e., on the way down)

    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)

    sol = solve_ivp(
        model_3d, [0, t_max_guess], initial_state,
        args=(m, c, g, wind_vec),
        events=hit_target_altitude, # <-- Use the modified event
        dense_output=False
    )
    
    if sol.status == 1 and len(sol.y_events[0]) > 0:
        return sol.y_events[0][-1]  # [x, y, z, vx, vy, vz] at impact
    else:
        # This can happen if the projectile never reaches target_z
        # (e.g., target_z = 100m but headwind is too strong)
        return None

# --- 4. Main Solver Function (Unchanged) ---
def find_launch_angles_3d(rho, wind_vector, target_distance, target_altitude_diff, v0, 
                          m_projectile=5.0, C_D=0.92, projectile_diameter=0.11, g=9.81,
                          verbose=True):
    target_z = target_altitude_diff
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * rho * A
    args_tuple = (v0, m_projectile, c_drag, g, wind_vector, target_distance, target_z)
    initial_guess = [10.0, 0.0] 
    
    if verbose:
        print(f"Solving for: Target X={target_distance}m, Z={target_z}m, Wind={wind_vector}")
    
    sol = root(objective_function_3d, initial_guess, args=args_tuple, method='lm')
    
    if sol.success:
        if verbose:
            print(f"--- ✅ Solution Successful: Theta={sol.x[0]:.4f}°, Phi={sol.x[1]:.4f}°")
        return sol.x[0], sol.x[1]
    else:
        if verbose:
            print(f"--- ❌ Solution Failed for X={target_distance}, Z={target_z}, Wind={wind_vector}")
        return None, None

# --- 5. 3D Trajectory Plotting (Unchanged) ---
def plot_trajectory_3d(theta_deg, phi_deg, v0, m, c, g, wind_vec, target_x, target_z, title_suffix=""):
    # (This function is unchanged)
    print(f"\n--- Generating 3D Trajectory Plot (Theta={theta_deg:.2f}°, Phi={phi_deg:.2f}°) ---")
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    v0z = v0 * np.sin(theta_rad)
    v_proj_xy = v0 * np.cos(theta_rad)
    v0x = v_proj_xy * np.cos(phi_rad)
    v0y = v_proj_xy * np.sin(phi_rad)
    initial_state = [0, 0, 0, v0x, v0y, v0z]
    
    # We'll plot until it hits z=0 (ground) for visualization
    def hit_ground(t, state, *args):
        return state[2]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    t_max_guess = max(30.0, (2.0 * v0 / g) * 1.5)
    sol = solve_ivp(
        model_3d, [0, t_max_guess], initial_state,
        args=(m, c, g, wind_vec),
        events=hit_ground, dense_output=True
    )
    if sol.t.size == 0:
        print("Error: Simulation failed to run.")
        return
    t_plot = np.linspace(0, sol.t[-1], 300)
    state_plot = sol.sol(t_plot)
    x_plot = state_plot[0]
    y_plot = state_plot[1]
    z_plot = state_plot[2]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_plot, y_plot, z_plot, label='Projectile Trajectory', color='blue', linewidth=2)
    ax.plot(x_plot, y_plot, 0, ':', color='grey', label='Ground Projection (Z=0)')
    ax.scatter(0, 0, 0, color='green', s=100, label='Launch Point (0,0,0)', depthshade=True)
    ax.scatter(target_x, 0, target_z, color='red', s=100, marker='X', label=f'Target (z={target_z}m)', depthshade=True)
    # (Rest of plotting unchanged)
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


# --- 7. Wind Sensitivity Analysis Table (MODIFIED) ---
def generate_wind_sensitivity_table(target_x, target_z, v0, m_projectile, c_drag, g, 
                                    wx_range=range(-10, 11), wy_range=range(0, 11), air_density=1.17,
                                    height_label=""):
    """
    Generate a table showing the impact of different wind conditions on targeting accuracy
    *** MODIFIED LOGIC ***
    This function calculates errors based on the impact state at z=target_z
    """
    print("\n" + "="*60)
    print(f"WIND SENSITIVITY ANALYSIS - Height: {target_z}m {height_label}")
    print("="*60)
    
    # 1. Calculate no-wind solution as reference
    print("Calculating height-specific reference solution...")
    wind_none = [0.0, 0.0, 0.0]
    
    A = np.pi * (0.11 / 2)**2 # Assuming diameter 0.11m
    C_D_calc = (2 * c_drag) / (air_density * A)
    
    theta_ref, phi_ref = find_launch_angles_3d(
        rho=air_density,
        wind_vector=wind_none,
        target_distance=target_x,
        target_altitude_diff=target_z, # <-- Uses the specific target_z
        v0=v0,
        m_projectile=m_projectile,
        C_D=C_D_calc,
        projectile_diameter=0.11,
        g=g,
        verbose=False 
    )
    
    if theta_ref is None:
        print("Failed to calculate reference solution. Cannot proceed.")
        return None, None, None, None, None
    
    print(f"Reference angles for this run: Theta={theta_ref:.4f}°, Phi={phi_ref:.4f}°")
    
    # Initialize results table
    results = []
    
    print(f"\nCalculating wind effects (using calculate_impact_point, stops at z={target_z})...")
    
    # --- MODIFIED SIMULATION LOOP ---
    
    # Test Wx variations (Wy=0)
    for wx in tqdm(wx_range, desc="Wx variations"):
        wind_vec = [wx, 0.0, 0.0]
        
        # *** MODIFIED CALL ***
        # Pass target_z to the impact function
        impact_state = calculate_impact_point(theta_ref, phi_ref, v0, m_projectile, c_drag, g, wind_vec, target_z)
        
        if impact_state is not None:
            impact_x, impact_y, impact_z = impact_state[0], impact_state[1], impact_state[2]
            
            x_error = impact_x - target_x
            y_error = impact_y - 0.0
            
            results.append({
                'Wind Type': 'Wx',
                'Wind Speed (m/s)': wx,
                'X Error (m)': x_error,
                'Y Error (m)': y_error,
                'Z Error (m)': impact_z - target_z # This should be ~0
            })
    
    # Test Wy variations (Wx=0)
    for wy in tqdm(wy_range, desc="Wy variations"):
        wind_vec = [0.0, wy, 0.0]
        
        # *** MODIFIED CALL ***
        # Pass target_z to the impact function
        impact_state = calculate_impact_point(theta_ref, phi_ref, v0, m_projectile, c_drag, g, wind_vec, target_z)
        
        if impact_state is not None:
            impact_x, impact_y, impact_z = impact_state[0], impact_state[1], impact_state[2]
            
            x_error = impact_x - target_x
            y_error = impact_y - 0.0

            results.append({
                'Wind Type': 'Wy',
                'Wind Speed (m/s)': wy,
                'X Error (m)': x_error,
                'Y Error (m)': y_error,
                'Z Error (m)': impact_z - target_z # This should be ~0
            })
    
    # Display results table
    print("\n" + "="*80)
    print(f"WIND SENSITIVITY ANALYSIS RESULTS - Height: {target_z}m {height_label}")
    print("="*80)
    print(f"Target: X={target_x}m, Z={target_z}m | Reference angles used: Theta={theta_ref:.2f}°, Phi={phi_ref:.2f}°")
    print(f"\nWind Effects Table (Errors measured at impact, z={target_z}m):")
    print("-" * 60)
    print(f"{'Wind Type':<10} {'Speed (m/s)':<12} {'X Error (m)':<12} {'Y Error (m)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['Wind Type']:<10} {result['Wind Speed (m/s)']:<12} {result['X Error (m)']:<12.2f} "
              f"{result['Y Error (m)']:<12.2f}")
    
    # Create visualizations
    wx_results = [r for r in results if r['Wind Type'] == 'Wx']
    wy_results = [r for r in results if r['Wind Type'] == 'Wy'] # Corrected typo
    
    return results, wx_results, wy_results, theta_ref, phi_ref


# --- 8. Linear Regression Analysis (Unchanged) ---
# This will now analyze the new X Error and Y Error data
# The intercepts should all be ~0
def perform_linear_regression_analysis(all_results, target_x):
    """
    Perform linear regression analysis on wind error data
    Analyzes X Error for Wx and Y Error for Wy
    """
    print("\n" + "="*80)
    print("LINEAR REGRESSION ANALYSIS: Wind Speed vs Error (Corrected Logic)")
    print("="*80)
    
    regression_results = {}
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    heights = list(all_results.keys())
    
    for height in heights:
        if height in all_results and all_results[height]['wx_results'] and all_results[height]['wy_results']:
            print(f"\n--- Height: {height}m ---")
            
            # Wx regression (head/tail wind vs RANGE error)
            wx_speeds = np.array([r['Wind Speed (m/s)'] for r in all_results[height]['wx_results']]).reshape(-1, 1)
            wx_x_errors = np.array([r['X Error (m)'] for r in all_results[height]['wx_results']])
            
            # Wy regression (cross wind vs LATERAL error)
            wy_speeds = np.array([r['Wind Speed (m/s)'] for r in all_results[height]['wy_results']]).reshape(-1, 1)
            wy_y_errors = np.array([r['Y Error (m)'] for r in all_results[height]['wy_results']])
            
            wx_model = LinearRegression()
            wx_model.fit(wx_speeds, wx_x_errors)
            wx_r2 = r2_score(wx_x_errors, wx_model.predict(wx_speeds))
            
            wy_model = LinearRegression()
            wy_model.fit(wy_speeds, wy_y_errors)
            wy_r2 = r2_score(wy_y_errors, wy_model.predict(wy_speeds))
            
            regression_results[height] = {
                'wx_slope': wx_model.coef_[0],
                'wx_intercept': wx_model.intercept_,
                'wx_r2': wx_r2,
                'wy_slope': wy_model.coef_[0],
                'wy_intercept': wy_model.intercept_,
                'wy_r2': wy_r2
            }
            
            print(f"Wx (Head/Tail Wind):")
            print(f"  Slope: {wx_model.coef_[0]:.4f} m/(m/s) (Range error per m/s wind)")
            print(f"  Intercept: {wx_model.intercept_:.4f} m (Error at Wx=0)")
            print(f"  R²: {wx_r2:.4f}")
            
            print(f"Wy (Cross Wind):")
            print(f"  Slope: {wy_model.coef_[0]:.4f} m/(m/s) (Lateral error per m/s wind)")
            print(f"  Intercept: {wy_model.intercept_:.4f} m (Error at Wy=0)")
            print(f"  R²: {wy_r2:.4f}")
    
    # Create regression visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Wx regression lines (NOW PLOTS X ERROR)
    plt.subplot(2, 3, 1)
    for i, height in enumerate(heights):
        if height in regression_results:
            wx_range = np.array([-10, 10]).reshape(-1, 1)
            wx_pred = regression_results[height]['wx_slope'] * wx_range + regression_results[height]['wx_intercept']
            plt.plot(wx_range, wx_pred, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m (slope={regression_results[height]["wx_slope"]:.2f})')
    
    plt.xlabel('Wx Wind Speed (m/s)')
    plt.ylabel('X Error (Range Error, m)')
    plt.title('Linear Regression: Range Error vs Head/Tail Wind')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Wy regression lines (Unchanged)
    plt.subplot(2, 3, 2)
    for i, height in enumerate(heights):
        if height in regression_results:
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
    plt.bar(x_pos - width/2, wx_r2_values, width, label='Wx (X Error) R²', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, wy_r2_values, width, label='Wy (Y Error) R²', color='blue', alpha=0.7)
    plt.xlabel('Target Height (m)')
    plt.ylabel('R² Value')
    plt.title('Regression Goodness-of-Fit (R²) by Height')
    plt.xticks(x_pos, heights)
    plt.ylim(0.9, 1.01) 
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # (Rest of plots are analogous)
    
    plt.tight_layout()
    plt.show()
    
    return regression_results


# --- 9. Multi-Height Analysis Function (MODIFIED) ---
def multi_height_wind_analysis(target_x, v0, m_projectile, c_drag, g, 
                               heights=[-100, -50, 0, 50, 100], 
                               wx_range=range(-10, 11), wy_range=range(0, 11), air_density=1.17):
    """
    Perform wind sensitivity analysis for multiple target heights
    *** MODIFIED ***
    Plots X Error vs Wx and Y Error vs Wy, using the new logic
    """
    print("\n" + "="*80)
    print("MULTI-HEIGHT WIND SENSITIVITY ANALYSIS (Logic: Error at z=target_z)")
    print("="*80)
    
    all_results = {}
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, height in enumerate(heights):
        print(f"\n\n{'='*60}")
        print(f"ANALYZING HEIGHT: {height}m")
        print(f"{'='*60}")
        
        height_label = "(Below Launch)" if height < 0 else "(Same Level)" if height == 0 else "(Above Launch)"
        
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
    
    plt.figure(figsize=(15, 5))
    
    # --- PLOT 1 ---
    plt.subplot(1, 2, 1) # Changed to 1,2,1
    for i, height in enumerate(heights):
        if height in all_results and all_results[height]['wx_results']:
            wx_speeds = [r['Wind Speed (m/s)'] for r in all_results[height]['wx_results']]
            wx_x_errors = [r['X Error (m)'] for r in all_results[height]['wx_results']] # Plot X Error
            plt.plot(wx_speeds, wx_x_errors, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m')
    
    plt.xlabel('Wx Wind Speed (m/s)')
    plt.ylabel('X Error (Range Error, m)') 
    plt.title('Range Error vs Head/Tail Wind Speed\n(Comparison Across Heights)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # --- PLOT 2 ---
    plt.subplot(1, 2, 2) # Changed to 1,2,2
    for i, height in enumerate(heights):
        if height in all_results and all_results[height]['wy_results']:
            wy_speeds = [r['Wind Speed (m/s)'] for r in all_results[height]['wy_results']]
            wy_y_errors = [r['Y Error (m)'] for r in all_results[height]['wy_results']]
            
            plt.plot(wy_speeds, wy_y_errors, color=colors[i], linewidth=2, 
                     label=f'Height: {height}m')
    
    plt.xlabel('Wy Wind Speed (m/s)')
    plt.ylabel('Y Error (Lateral Error, m)') 
    plt.title('Lateral Error vs Cross Wind Speed\n(Comparison Across Heights)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return all_results

# --- 10. Azimuth Correction Analysis (Unchanged) ---
def calculate_azimuth_corrections(all_heights_results, v0, m_projectile, C_D, projectile_diameter, g, air_density, target_x):
    """
    This function remains correct. It calculates the *true* angle
    needed to hit the target under wind.
    """
    print("\n" + "="*103)
    print("AZIMUTH (PHI) CORRECTION ANALYSIS FOR CROSSWIND (Wy)")
    print("="*103)
    print(f"{'Height (m)':<12} {'Phi (No Wind)':<15} {'Phi (Wy=5 m/s)':<18} {'Correction (Wy=5)':<20} {'Phi (Wy=10 m/s)':<18} {'Correction (Wy=10)':<20}")
    print("-" * 103)
    
    correction_data = {}
    
    for height, data in all_heights_results.items():
        phi_ref = data['phi_ref'] # This is the correct, height-specific reference
        
        if phi_ref is None:
            print(f"{height:<12} {'N/A (No Ref)':<15}")
            continue
            
        target_z = height
        
        wind_y5 = [0.0, 5.0, 0.0]
        theta_y5, phi_y5 = find_launch_angles_3d(
            air_density, wind_y5, target_x, target_z, v0, 
            m_projectile, C_D, projectile_diameter, g,
            verbose=False 
        )
        
        wind_y10 = [0.0, 10.0, 0.0]
        theta_y10, phi_y10 = find_launch_angles_3d(
            air_density, wind_y10, target_x, target_z, v0, 
            m_projectile, C_D, projectile_diameter, g,
            verbose=False
        )
        
        correction_y5 = phi_y5 - phi_ref if phi_y5 is not None else np.nan
        correction_y10 = phi_y10 - phi_ref if phi_y10 is not None else np.nan
            
        correction_data[height] = {
            'phi_ref': phi_ref,
            'phi_y5': phi_y5,
            'correction_y5': correction_y5,
            'phi_y10': phi_y10,
            'correction_y10': correction_y10
        }
        
        print(f"{height:<12} {phi_ref:<15.4f} {phi_y5:<18.4f} {correction_y5:<20.4f} {phi_y10:<18.4f} {correction_y10:<20.4f}")
        
    print("-" * 103)
    return correction_data


# --- 11. Main Program (REVERTED) ---
if __name__ == "__main__":
    
    # --- 1. Input your parameters here ---
    air_density = 1.17 
    target_x = 1200.0 
    v_initial = 450.0 

    # --- 2. Other parameters ---
    m_projectile = 5.0 
    C_D = 0.92 
    projectile_diameter = 0.11
    g = 9.81 
    
    # 3. Calculate drag constant c
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * air_density * A
    
    # ----------------------------------
    
    print("="*60)
    print(f"MULTI-HEIGHT WIND SENSITIVITY ANALYSIS (Logic: Error at z=target_z)")
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
    regression_results = perform_linear_regression_analysis(all_results, target_x)
    
    # --- 12. Azimuth Correction Analysis ---
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
    print("ANALYSIS (Logic: Error at z=target_z) COMPLETE")
    print("="*80)