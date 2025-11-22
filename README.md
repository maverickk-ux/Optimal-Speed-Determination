# Optimal Speed and Fuel Cost Minimization Using Optimization Algorithms

This project implements and compares classical optimization methods to determine the optimal driving speed that minimizes total travel cost, which includes fuel cost and time cost. The model incorporates real-world parameters such as traffic conditions, number of passengers, road gradient, engine efficiency, and fuel price.

This project was developed as part of an Optimization Course Project.

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

The optimization problem remains **convex** and smooth, making it ideal for classical optimization methods such as Newton and BFGS.

---

# Mathematical Model

The cost function minimized is:

\[
f(v) = \text{FuelCost}(v) + \lambda (v - v_{\text{pref}})^2
\]

Where:

- **FuelCost(v)** includes baseline fuel consumption, engine capacity adjustment, and calibrated penalties.
- **\(v_{\text{pref}}\)** decreases in high traffic and increases with engine capacity.
- **\(\lambda\)** controls the strength of speed preference regularization.

Additionally:

- Fuel efficiency is highest in the mid-speed band (approximately 30–60% of speed range).
- Penalties are added when the optimal speed lies outside this efficient range.
- A target fuel calibration ensures outputs like:
  - 400cc @ tau=0 → 0.37 L
  - 400cc @ tau=1 → 0.45 L  

---

# Optimization Algorithms Implemented

### **1️. Steepest Descent (Gradient Descent)**
- Uses only first derivative  
- Slowest convergence  
- Good introductory baseline  

### **2️. Gradient Descent with Armijo Line Search**
- Adaptive step size  
- More stable and accurate than fixed-step GD  

### **3️. Newton's Method**
- Uses analytic first and second derivatives  
- Very fast convergence (3–6 iterations)  
- Most accurate  
- Serves as the **reference method**  

### **4️. Quasi-Newton Method (BFGS)**
- Does not require second derivative  
- Builds inverse Hessian approximation  
- After tuning, matches Newton’s performance  
- Converges in 2–5 iterations  

### **5️. Trust-Region Method**
- Robust for poor starting points  
- Controls steps using region radius  

---

# How To Use

This project provides **separate files for each optimization method**, as well as an upcoming **combined comparison script** that runs all methods together.

---

## 1. Running Individual Optimization Methods

Each method is implemented in its own Python file:

- `NewtonMethod_Final.py`
- `BFGS_Final.py`
- `GradientDescent.py`
- `GradientDescent_LineSearch.py`
- `TrustRegion.py`

All files accept the **same user inputs**, for example:

<img width="394" height="197" alt="image" src="https://github.com/user-attachments/assets/2d985043-b118-4443-816d-8a77dc0318ab" />

Running any of these files will execute the corresponding algorithm and output:

- Optimal speed  
- Fuel used  
- Fuel cost  
- Regularization penalty  
- Number of iterations  

This allows you to easily check and understand the performance of each method independently.

---

## 2. Combined Comparison File (Upcoming)

We are preparing a **combined comparison script** that:

- Imports all individual optimization method files  
- Runs each method on **2–3 predefined test cases**  
- Measures:
  - Convergence speed  
  - Fuel cost  
  - Number of iterations  
  - Stability  

It will generate a **summary table** and **final recommendation** on which method is best suited for this optimization problem.

---

## 3. Providing Inputs to the Combined File

The combined file (e.g., `CompareAllMethods.py`) will accept **the same inputs** as the individual methods:

<img width="229" height="256" alt="image" src="https://github.com/user-attachments/assets/61d23630-82e8-48fc-aa22-88da625b9970" />


Once entered, the script will:

- Automatically apply these inputs to every optimization method  
- Run each solver  
- Print a comparison summary showing:
  - Optimal speeds  
  - Fuel consumption  
  - Cost  
  - Iterations  
  - Execution time  

This allows you to objectively compare all methods under identical conditions.

---

# Performance Summary (To be updated)

The following table compares the performance of all three optimization methods implemented: **Newton**, **BFGS**, and **Trust-Region**.

| Metric | Newton Method | BFGS Method | Trust-Region Method |
|--------|----------------|-------------|----------------------|
| **Convergence Speed** | Very Fast | Fast | Moderate |
| **Derivatives Used** | Uses full analytic \( f', f'' \) | Uses only \( f' \) (Hessian approximated) | Uses \( f' \) + model-based Hessian approximation |
| **Accuracy** | Highest accuracy | Matches Newton after correction | Matches Newton for optimum value |
| **Iterations** | 3–6 iterations | 2–5 iterations | 4–10 iterations (depends on radius updates) |
| **Stability** | Excellent | High after tuning | Very High (robust to poor initial guesses) |
| **Handles Bounds** | Requires projection | Requires projection | Built-in radius control makes it naturally stable near bounds |
| **Best Use Case** | Smooth convex problems with reliable derivatives | When second derivative is expensive/unavailable | When initial guess is poor or gradient is noisy |

---

# Example Output (Newton Method)
<img width="901" height="553" alt="image" src="https://github.com/user-attachments/assets/ab1bf1bb-b2cc-4641-b1f1-21c4a60c747f" />

# Example Output (BFGS Method)
<img width="874" height="518" alt="image" src="https://github.com/user-attachments/assets/3a433354-a57d-4e02-895d-16c3b120d2e5" />

# Example Output (Trust Region Method)
<img width="698" height="538" alt="image" src="https://github.com/user-attachments/assets/2be47297-96cb-4efa-ad9d-60970bc69ec5" />

 
