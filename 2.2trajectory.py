import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# --- 1. Physical Model (Differential Equations) ---
# (Unchanged, this was correct)
def model(t, state, m, c, g):
    """
    Defines the system of differential equations for projectile motion (f=cv^2 model).
    'state' is a list [x, y, vx, vy]
    """
    x, y, vx, vy = state
    v = (vx**2 + vy**2)**0.5
    
    # Avoid division by zero if v=0
    if v < 1e-9:
        return [0, 0, 0, -g]
        
    ax = -(c / m) * v * vx
    ay = -g - (c / m) * v * vy
    return [vx, vy, ax, ay]

# --- 2. "Forward" Simulator (Given angle, calculate trajectory) ---
# *** THIS FUNCTION IS HEAVILY MODIFIED AND CORRECTED ***
def get_y_at_x_target(theta_rad, v0, m, c, g, x_target):
    """
    Given a launch angle, simulate the trajectory and return the height y
    at horizontal distance x_target.
    
    This corrected version stops the simulation AT x_target,
    allowing it to solve for negative target heights.
    """
    
    # 1. Set initial conditions
    if np.cos(theta_rad) < 1e-9: # Handle near-vertical launch
        return -1e9
        
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)
    initial_state = [0, 0, v0x, v0y]
    
    # 2. Set simulation stop event (when x = x_target)
    # *** THIS IS THE CRITICAL LOGIC FIX ***
    def stop_at_target_x(t, state, m, c, g):
        return state[0] - x_target # Stop when x-coordinate = x_target
    
    stop_at_target_x.terminal = True  # Stop simulation on this event
    stop_at_target_x.direction = 1    # Trigger only when x is increasing
    
    # 3. Estimate a safe maximum simulation time
    # (Your robust method is good and unchanged)
    t_vacuum_max_flight = (2.0 * v0 / g) * 1.5 # 1.5x safety factor
    t_max_guess = max(30.0, t_vacuum_max_flight) # Ensure at least 30s
    
    
    # 4. Run ODE solver
    sol = solve_ivp(
        model,
        [0, t_max_guess],
        initial_state,
        args=(m, c, g),
        events=stop_at_target_x, # <-- Use the new, correct event
        dense_output=False      # We no longer need interpolation
    )
    
    # 5. Check results and return height
    # *** THIS LOGIC IS NOW SIMPLER AND CORRECT ***
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Event triggered: x_target was reached
        final_state = sol.y_events[0][-1]
        y_at_x = final_state[1] # This is the height at x_target
        return y_at_x
    else:
        # Simulation failed to reach x_target (e.g., out of range)
        # This is a "miss", return a huge error for the solver
        return -1e9 

# --- 3. "Inverse" Solver (Root Finder) ---
# (Unchanged, it now works correctly because get_y_at_x_target is fixed)
def find_launch_angle(x_target, y_target, v0, m, c, g=9.81, plot_solution=False, verbose=True, search_bracket=None):
    """
    Find the launch angle theta that hits (x_target, y_target).
    """
    
    if verbose:
        print("--- Starting Solution Process ---")
        print(f"Target: X={x_target} m, Y={y_target} m")
        print(f"Parameters: v0={v0} m/s, m={m} kg, c={c:.6f}")
    
    def objective_func(theta_deg):
        theta_rad = np.radians(theta_deg)
        y_at_x = get_y_at_x_target(theta_rad, v0, m, c, g, x_target)
        
        # Check if a valid float was returned
        if not np.isfinite(y_at_x):
            if verbose:
                print(f"  [Test] Angle: {theta_deg:.3f}°, FAILED (non-finite result)")
            return 1e10 if y_target < 0 else -1e10 
            
        error = y_at_x - y_target
        
        if verbose:
            print(f"  [Test] Angle: {theta_deg:.3f}°, Impact height: {y_at_x:.2f} m, Error: {error:.2f} m")
        return error

    solution_angle = None
    
    if search_bracket is not None:
        try:
            sol = root_scalar(objective_func, bracket=search_bracket, method='brentq')
            if sol.converged:
                solution_angle = sol.root
        except ValueError:
            pass # No root found in this bracket
    else:
        try: # Try low-angle solution first
            bracket_low = [0.1, 45.0]
            sol = root_scalar(objective_func, bracket=bracket_low, method='brentq')
            if sol.converged:
                solution_angle = sol.root
                if verbose: print(f"\n✅ Found low-angle solution!")
        except ValueError:
            pass

        if solution_angle is None: # If low-angle fails, try high-angle
            try:
                bracket_high = [45.0, 89.9]
                sol = root_scalar(objective_func, bracket=bracket_high, method='brentq')
                if sol.converged:
                    solution_angle = sol.root
                    if verbose: print(f"\n✅ Found high-angle solution!")
            except ValueError:
                pass
            
    if solution_angle is not None:
        if verbose:
            print(f"--- Solution Successful ---")
            print(f"Required launch angle: {solution_angle:.4f} degrees")
        if plot_solution:
            plot_trajectory(solution_angle, v0, m, c, g, x_target, y_target)
        return solution_angle
    else:
        if verbose:
            print(f"--- Solution Failed ---")
            print("❌ Error: Could not find solution. Target may be beyond maximum range.")
        return None

# --- 4. Plotting Function ---
# (Unchanged, this function was correct for plotting)
def plot_trajectory(theta_deg, v0, m, c, g, xt, yt):
    """
    Plot the final projectile trajectory.
    """
    print("... Generating trajectory plot ...")
    theta_rad = np.radians(theta_deg)
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)
    initial_state = [0, 0, v0x, v0y]
    
    def hit_ground(t, state, *args): return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    if v0x < 1e-9:
        t_max = (2.0 * v0 / g) * 1.5 # Use new robust estimate
    else:
        t_max = max((xt / v0x) * 2 + 5, (2.0 * v0 / g) * 1.5) # Use the safer of the two
    
    sol = solve_ivp(
        model, [0, t_max], initial_state, args=(m, c, g),
        events=hit_ground, dense_output=True
    )
    
    t_plot = np.linspace(0, sol.t[-1], 200)
    state_plot = sol.sol(t_plot)
    
    x_plot = state_plot[0]
    y_plot = state_plot[1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label='Projectile Trajectory (f=cv²)')
    plt.plot(xt, yt, 'ro', label=f'Target Point ({xt}, {yt})')
    
    # Only plot no-drag trajectory if v0y > 0
    if v0y > 0:
        t_no_drag = np.linspace(0, 2 * v0y / g, 200)
        x_no_drag = v0x * t_no_drag
        y_no_drag = v0y * t_no_drag - 0.5 * g * t_no_drag**2
        plt.plot(x_no_drag, y_no_drag, 'g--', label='No-Drag Trajectory (Vacuum)')
    
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Ballistic Trajectory (v0={v0} m/s, $\\theta$={theta_deg:.2f}°, m={m} kg, c={c:.6f})')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Adjust plot limits
    plt.axis('equal') 
    plt.ylim(bottom=min(0, np.min(y_plot), yt) - 5)
    # Ensure X-axis includes the target
    plt.xlim(left=min(0, xt) - 5, right=max(np.max(x_plot), xt) + 5)
    plt.show()

# --- 5. Parameter Analysis Plugin ---
# (Unchanged, it now works correctly because find_launch_angle is fixed)
def analyze_trajectory_parameters(x_target_fixed=1200, distance_range=(1000, 1500), 
                                  height_range=(-100, 100), v0=450, m=5, g=9.81):
    """
    Analyze how launch angle varies with different target parameters.
    This version shows both low and high angle solutions when available.
    """
    
    # Parameters
    C_D = 0.92
    rho = 1.17 
    A = np.pi * (0.11 / 2)**2
    c_drag = 0.5 * C_D * rho * A
    
    print("=== Trajectory Parameter Analysis ===")
    print(f"Fixed parameters: v0={v0} m/s, m={m} kg, c={c_drag:.6f}")
    
    bracket_low_angle = [0.1, 45.0]
    bracket_high_angle = [45.0, 89.9]

    # Part 1: Fixed distance (1200m), varying height
    print(f"\n--- Part 1: Fixed Distance ({x_target_fixed}m), Varying Height ---")
    heights = np.linspace(height_range[0], height_range[1], 50)
    
    results_low_angle = []
    results_high_angle = []
    
    for height in tqdm(heights, desc="Analyzing Heights (Plot 1)"):
        angle_low = find_launch_angle(
            x_target_fixed, height, v0, m, c_drag, g, 
            plot_solution=False, verbose=False, 
            search_bracket=bracket_low_angle
        )
        results_low_angle.append((height, angle_low))

        angle_high = find_launch_angle(
            x_target_fixed, height, v0, m, c_drag, g, 
            plot_solution=False, verbose=False, 
            search_bracket=bracket_high_angle
        )
        results_high_angle.append((height, angle_high))
            
    df_low = pd.DataFrame(results_low_angle, columns=['Height (m)', 'Launch Angle (deg)'])
    df_high = pd.DataFrame(results_high_angle, columns=['Height (m)', 'Launch Angle (deg)'])
    
    # Plot 1: Launch angle vs height at fixed distance
    plt.figure(figsize=(10, 6))
    
    valid_data_low = df_low.dropna(subset=['Launch Angle (deg)'])
    valid_data_high = df_high.dropna(subset=['Launch Angle (deg)'])

    plt.plot(valid_data_low['Height (m)'], valid_data_low['Launch Angle (deg)'], 
             'b-', linewidth=2, label='Low Angle Solution')
    plt.plot(valid_data_high['Height (m)'], valid_data_high['Launch Angle (deg)'], 
             'r--', linewidth=2, label='High Angle Solution')
    
    plt.xlabel('Target Height (m)')
    plt.ylabel('Launch Angle (degrees)')
    plt.title(f'Launch Angle vs Target Height\n(Fixed Distance: {x_target_fixed}m, v0={v0}m/s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Part 2: Fixed heights, varying distance
    print("\n--- Part 2: Fixed Heights, Varying Distance ---")
    distances = np.linspace(distance_range[0], distance_range[1], 50)
    fixed_heights = [-100, -50, 0, 50, 100]
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    all_low_angle_data = {}
    all_high_angle_data = {}
    
    for i, height in enumerate(fixed_heights):
        results_low_angle_dist = []
        results_high_angle_dist = []
        
        print(f"\nAnalyzing height = {height}m")
        
        for distance in tqdm(distances, desc=f"  Distances at y={height}m", leave=False):
            angle_low = find_launch_angle(
                distance, height, v0, m, c_drag, g, 
                plot_solution=False, verbose=False,
                search_bracket=bracket_low_angle
            )
            results_low_angle_dist.append((distance, angle_low))

            angle_high = find_launch_angle(
                distance, height, v0, m, c_drag, g, 
                plot_solution=False, verbose=False,
                search_bracket=bracket_high_angle
            )
            results_high_angle_dist.append((distance, angle_high))
        
        df_low_dist = pd.DataFrame(results_low_angle_dist, columns=['Distance (m)', 'Launch Angle (deg)'])
        df_high_dist = pd.DataFrame(results_high_angle_dist, columns=['Distance (m)', 'Launch Angle (deg)'])
        
        all_low_angle_data[height] = df_low_dist
        all_high_angle_data[height] = df_high_dist
    
    # Plot 2a: Low Angle Solutions (0-20 degrees)
    plt.figure(figsize=(12, 8))
    
    for i, height in enumerate(fixed_heights):
        df_low_dist = all_low_angle_data[height]
        valid_data_low_dist = df_low_dist.dropna(subset=['Launch Angle (deg)'])
        
        low_angle_mask = valid_data_low_dist['Launch Angle (deg)'] <= 20
        valid_low_angles = valid_data_low_dist[low_angle_mask]
        
        plt.plot(valid_low_angles['Distance (m)'], valid_low_angles['Launch Angle (deg)'], 
                 color=colors[i], marker='.', markersize=6, linestyle='-', 
                 linewidth=2, label=f'Height = {height}m (Low Angle)')
    
    plt.xlabel('Target Distance (m)')
    plt.ylabel('Launch Angle (degrees)')
    plt.title(f'Low Angle Solutions (0-20°)\n(Fixed Heights, v0={v0}m/s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.show()
    
    # Plot 2b: High Angle Solutions (55-75 degrees)
    plt.figure(figsize=(12, 8))
    
    for i, height in enumerate(fixed_heights):
        df_high_dist = all_high_angle_data[height]
        valid_data_high_dist = df_high_dist.dropna(subset=['Launch Angle (deg)'])
        
        high_angle_mask = (valid_data_high_dist['Launch Angle (deg)'] >= 55) & \
                          (valid_data_high_dist['Launch Angle (deg)'] <= 75)
        valid_high_angles = valid_data_high_dist[high_angle_mask]
        
        plt.plot(valid_high_angles['Distance (m)'], valid_high_angles['Launch Angle (deg)'], 
                 color=colors[i], marker='x', markersize=6, linestyle='--', 
                 linewidth=2, label=f'Height = {height}m (High Angle)')
    
    plt.xlabel('Target Distance (m)')
    plt.ylabel('Launch Angle (degrees)')
    plt.title(f'High Angle Solutions (55-75°)\n(Fixed Heights, v0={v0}m/s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(55, 75)
    plt.tight_layout()
    plt.show()
    
    # Print summary tables
    print("\n=== Summary Table ===")
    print(f"Angle vs Height @ {x_target_fixed}m (Low Angle):")
    print(df_low.to_string(index=False, float_format="%.2f"))
    print(f"\nAngle vs Height @ {x_target_fixed}m (High Angle):")
    print(df_high.to_string(index=False, float_format="%.2f"))
    
    return df_low, df_high, all_low_angle_data, all_high_angle_data

# --- 6. Main Program ---
if __name__ == "__main__":
    
    # --- Main parameters ---
    m_projectile = 5.0
    v_launch     = 450.0
    
    # --- Single target test ---
    target_x   = 1200.0
    target_y   = 0.0
    
    # --- Drag Constant Calculation ---
    C_D = 0.92
    rho = 1.17
    projectile_diameter = 0.11 
    A = np.pi * (projectile_diameter / 2)**2
    c_drag = 0.5 * C_D * rho * A
    
    # ----------------------------------
    
    print("="*60)
    print("SINGLE TARGET ANALYSIS (Target: 1200m, 0m)")
    print("="*60)
    angle = find_launch_angle(
        target_x,
        target_y,
        v_launch,
        m_projectile,
        c_drag,
        plot_solution=True,
        verbose=True
    )

    print("\n" + "="*60)
    print("SINGLE TARGET ANALYSIS (Target: 1200m, -100m)")
    print("="*60)
    # Test a negative height target to prove the fix works
    angle_neg = find_launch_angle(
        target_x,
        -100.0,
        v_launch,
        m_projectile,
        c_drag,
        plot_solution=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("BATCH PARAMETER ANALYSIS")
    print("="*60)
    
    # *** THIS IS THE CORRECTED LINE ***
    # It now correctly unpacks all 4 return values
    df_low_height, df_high_height, all_low_angle_data, all_high_angle_data = analyze_trajectory_parameters(
        x_target_fixed=1200,
        distance_range=(1000, 1500),
        height_range=(-100, 100),
        v0=v_launch,
        m=m_projectile
    )