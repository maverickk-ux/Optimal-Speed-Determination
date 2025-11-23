# Optimal Speed and Fuel Cost Minimization Using Optimization Algorithms

This project implements and compares classical optimization methods to determine the optimal driving speed that minimizes total travel cost, which includes fuel cost and time cost. The model incorporates real-world parameters such as traffic conditions, number of passengers, road gradient, engine efficiency, and fuel price.

This project was developed as part of an Optimization Course Project.

 # Team: BP
- K V Shashank Pai (BT2024250)
- Varshith M Gowda (BT2024227)
- Muppana Jatin (BT2024127)

## Project Overview
Commuters face a trade-off between fuel efficiency and travel time. Driving too slowly increases travel time, while driving too fast increases fuel consumption due to aerodynamic drag.

This project models that trade-off and uses optimization techniques to find the speed \(v\) (in km/h) that results in the minimum total trip cost.

---

# Project Description

To achieve realistic optimization, the model integrates:

- Traffic-dependent preferred speed  
- Engine capacity influence on speed and fuel usage  
- Passenger load impact  
- Speed limits (min/max)  
- A calibrated fuel model with penalty terms outside the efficient speed range  

A regularization term ensures the final speed is not too far from the traffic-adjusted desired speed.

The optimization problem remains **convex** and smooth, making it ideal for classical optimization methods such as Newton, Trust region method and BFGS.

---

# Mathematical Model

The cost function minimized is:

$$
f(v) = \text{FuelCost}(v) + \lambda (v - v_{\text{pref}})^2
$$

Where:

- **FuelCost(v)** includes baseline fuel consumption, engine capacity adjustment, and calibrated penalties.
- **$( v_{\text{pref}} \)$** decreases in high traffic and increases with engine capacity.
- **$( \lambda \)$** controls the strength of speed preference regularization.

Additionally:

- Fuel efficiency is highest in the mid-speed band (approximately **30–60%** of the speed range).
- Penalties are added when the optimal speed lies outside this efficient range.
- A target fuel calibration ensures outputs like:
  - **400cc @ tau = 0 → 0.37 L , for D=10km, n=1**
  - **400cc @ tau = 1 → 0.45 L , for D=10km, n=1**

---

# Optimization Algorithms Implemented

### **1️. Steepest Descent (Gradient Descent)**
- Uses only first derivative  
- Slowest convergence  
- Good introductory baseline  

### **2. Newton's Method**
- Uses analytic first and second derivatives  
- Very fast convergence (3–6 iterations)  
- Most accurate  
- Serves as the **reference method**  

### **3. Quasi-Newton Method (BFGS)**
- Does not require second derivative  
- Builds inverse Hessian approximation  
- After tuning, matches Newton’s performance  
- Converges in 2–5 iterations  

### **4. Trust-Region Method**
- Robust for poor starting points  
- Controls steps using region radius  

---

# How To Use the Optimization Methods

This project includes separate Python files for each optimization algorithm.  
All four methods use the **same input format**, making it easy to compare their performance.

---

## 1. Running Each Optimization Method Separately

Each optimization method is implemented in its own file:

### Files
- `NewtonMethod.py`
- `BFGS.py`
- `SteepestDescent.py`
- `TrustRegion.py`

Each file takes the following inputs through the terminal:

<img width="394" height="197" alt="image" src="https://github.com/user-attachments/assets/2d985043-b118-4443-816d-8a77dc0318ab" />


### What each script outputs:
- Optimal speed $\( v^* \)$  
- Total fuel used  
- Fuel cost  
- Regularization penalty  
- Total fuel cost at optimal speed  
- Number of iterations  
- Method-specific diagnostics (Newton gradient, BFGS updates, TR radius, etc.)

### Example Output (Newton Method)
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/d811e5ff-48dc-49f6-8059-6fb06a0cce1b" />

### Example Output (BFGS Method)
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/52f2444a-8678-466f-a902-db72f926f4c0" />

### Example Output (Trust Region Method)
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/665edcd5-5b69-42b1-a2a7-796b3fb9a148" />

### Example Output (Steepest Descent)
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/cd4ad1a1-2dd1-425a-97da-7b36ee9bd29f" />

All four methods converge to the **same optimal speed**, proving correctness and consistency.

---

## 2. Combined Comparison Script – Final Implementation

We implemented a single integrated Python script, **`CompareAllMethods.py`**, which runs and compares all four optimization methods on the same cost function and inputs:

- **Steepest Descent** (Armijo backtracking)  
- **Newton’s Method**  
- **BFGS (quasi-Newton)**  
- **Trust-Region Method**

### 2.1 What the Script Does

For a given set of user inputs:

- Distance \(D\)  
- Traffic index \(\tau\)  
- Number of passengers \(n\)  
- Engine capacity \(C\) (cc)  
- Speed bounds \([v_{\min}, v_{\max}]\)  
- Initial guess \(v_0\)

the script:

1. Builds a **common objective function**
   \[
   f(v) = \text{FuelCost}(v) + \text{TimeCost}(v) + \lambda_{\text{reg}}\,(v - v_{\text{pref}})^2
   \]
   so all methods optimize exactly the same model.

2. Runs **all four methods** from the same initial guess \(v_0\) with box constraints enforced via projection.

3. Prints a **detailed result block** for each method:
   - Number of iterations  
   - Optimal speed \(v^\*\) (km/h)  
   - Total fuel used (L)  
   - Fuel cost  
   - Regularization penalty  
   - Total cost (fuel + penalty)

4. Produces a **comparison table** summarizing:
   - Method  
   - Iteration count  
   - Optimal speed  
   - Fuel consumption  
   - Total cost  

This allows direct comparison of accuracy and efficiency across all algorithms.

### 2.2 Plots Generated

The script also generates a set of plots for visual analysis.

#### (a) 2×2 Method Comparison Dashboard

A 2×2 subplot shows:

1. **Iterations per method**  
2. **Optimal speed  $\(v^\*\)$**  
3. **Total fuel used**  
4. **Total cost (fuel + regularization)**  

These bar charts make it easy to see which method converges fastest and whether all methods agree on the final solution.

#### (b) Convergence-Diagnostics Plots

Using per-iteration histories for **Steepest Descent, Newton, BFGS, and Trust Region**, the script plots:

1. **Gradient norm vs iteration** (log scale)  
2. **Step magnitude vs iteration** (log scale)  
3. **Objective value vs iteration**  
4. **Objective $\(f(v)\)$ with method trajectories in $\(v\)$-space**  
   - Yellow curve: continuous objective \(f(v)\)  
   - Markers: iterates from each method moving toward the minimum  

These diagnostic plots show how quickly each method converges, how aggressively it updates the speed, and how their paths differ while still reaching essentially the same optimal speed.

### 2.3 How to Run

python CompareAllMethods.py

## 3. Providing Inputs to the Combined File

`CompareAllMethods.py` will accept **exactly the same inputs** as the individual files:

<img width="229" height="256" alt="image" src="https://github.com/user-attachments/assets/61d23630-82e8-48fc-aa22-88da625b9970" />



After entering them only once, the script will:

- Pass the inputs to all optimization methods  
- Run them sequentially  
- Display a combined, aligned comparison report  

---

# Summary of Current Method Behaviour

| Metric | Newton | BFGS | Steepest Descent | Trust-Region |
|--------|--------|------|------------------|--------------|
| Convergence Speed | Very Fast | Fast | Moderate | Fast |
| Iterations (typical) | 4–6 | 6–10 | 10–15 | 4–8 |
| Derivatives Used | \( f', f'' \) | \( f' \) only | \( f' \) only | \( f', f'' \) (model) |
| Accuracy | Exact | Matches Newton | Matches Newton | Matches Newton |
| Robustness | High | High | Medium | Very High |
| Best Use Case | Clean convex problems | When Hessian not available | Simple baseline solver | Poor initial guesses / unstable gradients |

---
