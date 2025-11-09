# Project 1
The main purpose of this project is to complete some necessary numerical calculations arising from the non-analytic behavior of differential equations in Problem B of the UPC 2025 that we are working on.
The project goes beyond simple trajectory calculation and includes:
1.  **Model Validation:** Verifies the accuracy of the physical model against historical firing table data from the "Napoleon" field gun.
2.  **3D Ballistic Solution:** Solves the inverse problem by calculating the precise launch elevation (Theta) and azimuth (Phi) required to hit a 3D target under given wind conditions.
3.  **Sensitivity Analysis:** Performs a quantitative analysis to measure the linear impact of wind (head/tail wind and cross wind) on the projectile's point of impact at various target altitudes.
4.  **Visualization:** Generates 2D error analysis charts and full 3D trajectory plots.


## Project Sturcture
```bash
ProblemB/
│  README.md                 # Project documentation
│  2.2example.py             # Using Napoleon cannon data to validate the algorithm
│  2.2trajectory.py          # Main script for training and evaluation
│  2.3simplified.py          # Consider the two-dimensional wind speed toward a fixed target (the simplest case)
│  2.3trajectory.py          # Consider effect of wind speed at different heights, and incorporate linearization.
|  Machine Learning.ipynb    # Using pytorch to solve difficult w_x and h problem and simulate
│
├─figure                    # Visualization outputs
│ Angle_vs_Target_Height.png   
│ Hish_Angle_Solutions.png  
│ Low_Angle_Solutions.png  
│ Simulation_for_Napoleon.png  
│ Trajectory_with_viscosity.png  
│ 3D_Trajectory_Error.png  
│ 3D_Trajectory_with_Wind.png  
│ Head_Tail_and _Cross_Wind_Error.png  
│ Lateral_and_Horizontal_Wind_Error.png    
│
└─__pycache__               # Auto-generated cache files
```



---

## 🛠️ Requirements

This project requires Python 3.x and the following third-party libraries:

* `numpy`: For numerical calculations.
* `scipy`: For solving Ordinary Differential Equations (ODE) and finding numerical roots.
* `matplotlib`: For plotting 2D and 3D charts.
* `pandas`: For organizing and displaying data tables from the sensitivity analysis.
* `tqdm`: For displaying progress bars during batch calculations.
* `sklearn`: For solving some linear ansatz and inverse questions
* `torch`: For displaying machine learning for this difficult problem


You can install all dependencies using pip:
```bash
pip install numpy scipy matplotlib pandas tqdm scikit-learn
````

-----

## ⚛️ Core Physical Model

All simulations are based on a unified physical model adapted to your specified coordinate system.

### 3D Coordinate System Specification

  * `+X Axis`: **Forward** (Horizontal baseline towards the target)
  * `+Y Axis`: **Lateral Deviation** (Positive to the right, used for cross wind and azimuth `phi`)
  * `+Z Axis`: **Vertical Height** (Positive upwards)

### Forces Considered

1.  **Gravity ($F_g$):** Acts constantly in the `-Z` direction.
      * $a_z = -g$
2.  **Quadratic Air Drag ($F_d$):**
      * $F_d = -c \cdot |\vec{v}_{\text{rel}}| \cdot \vec{v}_{\text{rel}}$
      * $\vec{v}_{\text{rel}} = \vec{v}_{\text{projectile}} - \vec{w}_{\text{wind}}$
      * $c = 0.5 \cdot C_D \cdot \rho \cdot A$

### Solved Angles

  * **Elevation Angle (Theta, $\theta$):** The angle in the **X-Z plane** (vertical aim).
  * **Azimuth Angle (Phi, $\phi$):** The angle in the **X-Y plane** (horizontal aim).

-----

## 🚀 File Structure & Usage Guide

The project files are organized by function, progressing from basic validation to advanced analysis:

### 2.2example.py - Algorithm Validation

  * **Purpose:** The core goal of this script is to **validate** the physical model and solving algorithm.
  * **Method:** It loads historical firing table data (elevation, range) from the **M1857 12-pounder "Napoleon" field gun** and compares it to the ranges simulated by the physical model in `2.2trajectory.py`.
  * **How to Use:** `python 2.2example.py`

### 2.2trajectory.py - 3D Drag Effect Analysis

  * **Purpose:** A core library/script that provides a detailed analysis of how **air resistance** in three dimensions affects the projectile's trajectory.
  * **Method:** It contains the core 3D physical model (`model_3d`) and the forward simulator. It is likely **imported** by `2.2example.py` to perform the validation.
  * **How to Use:** This is most likely a library file imported by other scripts, not run directly.

### 2.3simplified.py - 3D Ballistic Solver (Single Point)

  * **Purpose:** This is a **main executable script** for solving a specific, concrete ballistic problem.
  * **Method:** For a **fixed target altitude (z)** and **fixed distance (x)**, this script calculates the **elevation (theta)** and **azimuth (phi)** required for a precise hit, accounting for a 3D wind vector.
  * **How to Use:**
    1.  Open this file and modify the parameters in the `if __name__ == "__main__":` block at the bottom.
    2.  Run the script: `python 2.3simplified.py`
  * **Output:** Prints the calculated `theta` and `phi` to the console and calls the 3D plotting function to visualize the trajectory.

### 2.3trajectory.py - Wind Sensitivity Analysis

  * **Purpose:** This is the project's core analysis tool, designed to **quantify** the effect of wind speed on accuracy.
  * **Method:** It uses the `generate_wind_sensitivity_table` and `multi_height_wind_analysis` functions to analyze how different wind speeds (Wx and Wy) cause **impact point errors** (X Error, Y Error) at different target altitudes.
  * **Core Logic:** It first calculates a "no-wind" aiming solution (`theta_ref`, `phi_ref`), then "fires" using those angles in a "windy" environment to measure how many meters the shot is off by.
  * **How to Use:**
    1.  Open this file and modify the analysis parameters in the `if __name__ == "__main__":` block.
    2.  Run the script: `python 2.3trajectory.py`
  * **Output:** Prints detailed error tables (including linear regression factors) and generates 2D comparison charts showing the relationship between wind speed and impact error.

-----

## 🔑 Key Parameters to Adjust

To configure a simulation, edit the `if __name__ == "__main__":` block in the main scripts (primarily `2.3simplified.py` or `2.3trajectory.py`):

  * `air_density` (or `rho`): Air density (kg/m³)
  * `wind_vector`: The 3D wind velocity `[wx, wy, wz]` (m/s)
      * `wx`: Head/Tail wind (+X / -X)
      * `wy`: Cross wind (+Y to the right / -Y to the left)
      * `wz`: Up/Down draft (+Z / -Z)
  * `target_x`: The horizontal distance to the target (m)
  * `target_z` (or `target_altitude_diff`): The vertical altitude difference to the target (m)
  * `v_initial` (or `v0`): The projectile's initial velocity (m/s)
  * `m_projectile`: The projectile's mass (kg)
  * `C_D`: The coefficient of drag (dimensionless)
  * `projectile_diameter`: The projectile's diameter (m) (used to calculate cross-sectional area $A$)

<!-- end list -->
