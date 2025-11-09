import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- 1. Physical Model (Differential Equations) ---

def model(t, state, m, c, g):
    """
    Define the system of differential equations for projectile motion (f=cv^2 model).
    state = [x, y, vx, vy]
    """
    # Unpack state variables
    x, y, vx, vy = state
    
    # Calculate instantaneous speed v = sqrt(vx^2 + vy^2)
    v = (vx**2 + vy**2)**0.5
    
    # f = c*v^2, f_x = f * (vx/v) = c*v*vx
    # f_y = f * (vy/v) = c*v*vy
    
    # Calculate acceleration
    # m * ax = -f_x
    ax = -(c / m) * v * vx
    
    # m * ay = -mg - f_y
    ay = -g - (c / m) * v * vy
    
    # Return derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
    return [vx, vy, ax, ay]

# --- 2. Function to calculate range for a given angle and height ---

def calculate_range(theta_deg, v0, m, c, g, initial_height=0):
    """
    Calculate the horizontal range for a given launch angle and initial height.
    Returns the x-coordinate when the projectile hits the ground (y=0).
    """
    theta_rad = np.radians(theta_deg)
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)
    initial_state = [0, initial_height, v0x, v0y]  # Added initial height
    
    # Event to stop simulation when projectile hits ground
    def hit_ground(t, state, m, c, g):
        return state[1]  # y coordinate
    
    hit_ground.terminal = True
    hit_ground.direction = -1  # Only trigger when y is decreasing
    
    # Estimate maximum simulation time
    t_max_guess = 100.0  # Generous upper bound
    
    # Run simulation
    sol = solve_ivp(
        model,
        [0, t_max_guess],
        initial_state,
        args=(m, c, g),
        events=hit_ground,
        dense_output=False
    )
    
    # Return the x-coordinate when projectile hits ground
    if sol.status == 1 and len(sol.y_events[0]) > 0:
        final_state = sol.y_events[0][-1]
        return final_state[0]  # x coordinate at impact
    else:
        # If no ground hit event, use the last point
        return sol.y[0][-1]

# --- 3. Function to find height that gives 274m range at 0° ---

def find_initial_height_for_0_degree_range(target_range, v0, m, c, g):
    """
    Find the initial height that results in the target range when launched at 0°.
    """
    def objective(height):
        range_val = calculate_range(0, v0, m, c, g, height)
        return range_val - target_range
    
    # Use root finding to determine the required height
    try:
        # Try a reasonable bracket for height (0 to 1000m)
        sol = root_scalar(objective, bracket=[0, 1000], method='brentq')
        if sol.converged:
            return sol.root
        else:
            return None
    except ValueError:
        # If bracket doesn't work, try a different approach
        heights = np.linspace(0, 1000, 100)
        ranges = [calculate_range(0, v0, m, c, g, h) for h in heights]
        
        # Find height that gives closest range to target
        idx = np.argmin(np.abs(np.array(ranges) - target_range))
        return heights[idx]

# --- 4. Main Program: Test multiple angles and plot results ---

if __name__ == "__main__":
    # Physical parameters (same as original)
    m_projectile = 7.2     # Projectile mass (kg)
    v_launch = 514         # Launch velocity (m/s)
    
    # Drag parameters
    C_D = 0.92 
    rho = 1.17             # Air density (kg/m³)
    A = np.pi * (0.13 / 2)**2  # Cross-sectional area (m²)
    c_drag = 0.5 * C_D * rho * A  # Drag constant
    
    g = 9.81  # Gravity (m/s²)
    
    # Find initial height that gives 274m range at 0°
    target_0_degree_range = 274  # meters
    print("--- Finding Initial Height for 0° Range = 274m ---")
    initial_height = find_initial_height_for_0_degree_range(
        target_0_degree_range, v_launch, m_projectile, c_drag, g
    )
    
    if initial_height is not None:
        print(f"Required initial height: {initial_height:.2f} m")
        
        # Verify the height gives the correct range at 0°
        verified_range = calculate_range(0, v_launch, m_projectile, c_drag, g, initial_height)
        print(f"Verified 0° range with height {initial_height:.2f} m: {verified_range:.2f} m")
    else:
        print("Could not find appropriate initial height")
        initial_height = 0
    
    print("\n--- Projectile Range vs Launch Angle Analysis ---")
    print(f"Parameters: v0={v_launch} m/s, m={m_projectile} kg, c={c_drag:.6f}, h={initial_height:.2f} m")
    print(f"Testing angles: 0°, 1°, 2°, 3°, 4°, 5°, 6°, 10°")
    print()
    
    # Angles to test
    test_angles = [0, 1, 2, 3, 4, 5, 6, 10]
    ranges = []
    
    # Calculate range for each angle with the determined initial height
    for angle in test_angles:
        print(f"Calculating range for {angle}°...")
        range_val = calculate_range(angle, v_launch, m_projectile, c_drag, g, initial_height)
        ranges.append(range_val)
        print(f"  {angle}° → Range: {range_val:.2f} m")
    
    # Create results table
    print("\n--- Results Summary ---")
    print("Angle (°) | Range (m)")
    print("----------|-----------")
    for angle, range_val in zip(test_angles, ranges):
        print(f"{angle:8}° | {range_val:9.2f} m")
    
    # --- Create the line plot ---
    plt.figure(figsize=(12, 8))
    
    # Line plot with markers
    plt.plot(test_angles, ranges, 'o-', linewidth=3, markersize=8, 
             color='steelblue', markerfacecolor='red', markeredgecolor='darkred', 
             markeredgewidth=2)
    
    # Add value labels on points
    for angle, range_val in zip(test_angles, ranges):
        plt.annotate(f'{range_val:.0f} m', 
                    xy=(angle, range_val), 
                    xytext=(5, 10), 
                    textcoords='offset points',
                    fontsize=11, 
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Highlight the 0° point with special annotation
    plt.annotate(f'0°: {ranges[0]:.0f} m (Target: 274m)', 
                xy=(test_angles[0], ranges[0]), 
                xytext=(20, 30), 
                textcoords='offset points',
                fontsize=12, 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Plot styling
    plt.xlabel('Launch Angle (degrees)', fontsize=14, fontweight='bold')
    plt.ylabel('Horizontal Range (m)', fontsize=14, fontweight='bold')
    plt.title(f'Projectile Range vs Launch Angle\n'
             f'(v₀ = {v_launch} m/s, m = {m_projectile} kg, c = {c_drag:.4f}, h = {initial_height:.2f} m)', 
             fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(test_angles)
    
    # Set y-axis limits to show data clearly
    y_min = min(ranges) * 0.8
    y_max = max(ranges) * 1.1
    plt.ylim(y_min, y_max)
    
    # Add some statistics
    max_range = max(ranges)
    max_angle = test_angles[ranges.index(max_range)]
    plt.axhline(y=max_range, color='red', linestyle=':', alpha=0.7, 
                label=f'Max Range: {max_range:.0f} m at {max_angle}°')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    # Additional analysis
    print(f"\n--- Analysis ---")
    print(f"Maximum range: {max_range:.2f} m at {max_angle}°")
    print(f"Range at 45° (theoretical optimum in vacuum): {calculate_range(45, v_launch, m_projectile, c_drag, g, initial_height):.2f} m")