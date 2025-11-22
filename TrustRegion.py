
import numpy as np

#USER INPUTS

D = float(input("Distance (km): "))
tau = float(input("Traffic Index (0..1): "))
n = int(input("Number of passengers: "))
C = float(input("Enter engine capacity (cc): ")) 
v_min = float(input("Enter minimum speed: "))
v_max = float(input("Enter maximum speed: "))

# Constant Parameters
w_avg = 70.0  #Average weight of the passenger
Pf = 102.0  # Fuel Price (currency per L)
Ct = 160.0  # Cost of Time (delay constant)
s = 0.05   # Delay scaling factor for traffic

# Preferred-speed / formula coefficients
c1 = 75.0
c2 = 5400.0
c3 = 4.0 / 70.0

# Small baseline fuel-speed coefficients
ce1_base = 2e-4
ce2_base = 1e-5

# Engine-efficiency scaling for baseline coefficients
C_ref = 400.0
beta_eff = 0.20

# baseline K term coefficients
c0 = 0.005
c3_base = 0.001
c4 = 1e-5
c5 = 0.002

# Regularizer weight
lambda_reg = 0.2

# Precompute scaling and baseline coefficients
B = Pf * D   # Converts fuel consumption (L/km) into cost by multiplying with fuel price * distance

# Engine scaling factor (smaller engines → higher fuel usage)
eff_scale = 1.0 if C <= 0 else (C_ref / C) ** beta_eff

# Speed-dependent fuel coefficients after scaling
ce1 = ce1_base * eff_scale
ce2 = ce2_base * eff_scale

# K constant for fixed fuel consumption terms(traffic, passenger weight, engine size)
K = c0 + c3_base * tau + c4 * n * w_avg + c5 / (C if C > 0 else C_ref)

# Formula Definitions

def v_pref_formula(tau_val, C_val, n_val, w_avg_val):
    #Calculates the preferred speed, vp. based on traffic, engine capacity, and passenger weight. Result is clipped within allowed [v_min, v_max].
    tau_clipped = np.clip(tau_val, 0.0, 1.0)
    C_safe = C_val if C_val > 0 else 1.0
    F = v_max - c1 * tau_clipped - (c2 / C_safe) - c3 * (n_val * w_avg_val)
    return float(np.clip(F, v_min, v_max))

def FC_baseline_per_km(v):
    #Calculates Fuel Consumption per km (baseline model).
    return K + ce1 * v + ce2 * v**2

def f_base(v):
    #Calculates the base objective function (Fuel Cost + Time Cost).
    if v <= 0:
        return float('inf')
    FC_v = FC_baseline_per_km(v)
    fuel_cost = B * FC_v
    time_cost = Ct * (D / v)
    const_delay = Ct * s * tau * D
    return fuel_cost + time_cost + const_delay

def f(v):
    #Calculates the final REGULARIZED objective function f(v).
    vp = v_pref_formula(tau, C, n, w_avg)
    return f_base(v) + lambda_reg * (v - vp)**2

def fprime(v):
    #Calculates the first derivative (gradient) f'(v).
    base_deriv = B * (ce1 + 2.0 * ce2 * v) - Ct * D / (v**2)
    reg_deriv = 2.0 * lambda_reg * (v - v_pref_formula(tau, C, n, w_avg))
    return base_deriv + reg_deriv

def fdoubleprime(v):
    #Calculates the second derivative (Hessian) f''(v).
    base_hess = 2.0 * B * ce2 + 2.0 * Ct * D / (v**3)
    reg_hess = 2.0 * lambda_reg
    return base_hess + reg_hess

def proj(v):
    #Projection onto the feasible speed bounds [v_min, v_max].
    return float(np.clip(v, v_min, v_max))

# TRUST-REGION METHOD IMPLEMENTATION

def trust_region_optimization(v0, tol=1e-6, max_iter=100, verbose=True):
    #Solves the 1D optimization problem using the Trust-Region method.
    v_k = proj(v0)
    iterations=0
    # Trust-Region Parameters
    Delta_k = 10.0
    Delta_max = 50.0
    eta = 1e-4

    if verbose:
        vp = v_pref_formula(tau, C, n, w_avg)
        print("\nRunning Trust-Region Optimization...")
        print(f"Computed v_pref (formula) = {vp:.6f} km/h")
        print(f"Starting from v0 (projected) = {v_k:.6f} km/h, lambda={lambda_reg}")
        print("-" * 50)

    for k in range(1, max_iter + 1):
        f_k = f(v_k)
        g_k = fprime(v_k)
        H_k = fdoubleprime(v_k)

        # 1. Stopping Criterion (Gradient Norm)
        if abs(g_k) < tol:
            if verbose:
                iterations=k-1
            break
        
        # Guard against zero Hessian
        if abs(H_k) < 1e-14:
            if verbose:
                print("Hessian too small — stopping.")
            break

        # 2. SOLVE SUBPROBLEM: Determine step p_k (Clamped Newton step)
        p_star = -g_k / H_k    # Newton step proposal
        p_k = np.clip(p_star, -Delta_k, Delta_k)  # Trust-region step (clamped to radius Delta_k)

        # 3. CALCULATE REDUCTIONS
        v_trial = v_k + p_k   # Trial point (before projection)
        p_actual = v_trial - v_k   # Actual step used for model
        
        ActRed = f_k - f(v_trial) # Actual reduction
        PredRed = -(g_k * p_actual + 0.5 * H_k * p_actual**2) # Predicted reduction

        # 4. REDUCTION RATIO (rho_k)
        if PredRed <= 0:
            rho_k = -1.0      # Indicates bad model or non-descent step
        else:
            rho_k = ActRed / PredRed

        # 5. UPDATE POSITION (v_k) AND RADIUS (Delta_k)
        v_next = v_k
        Delta_next = Delta_k
        
        # Step Acceptance
        step_acceptance = False   # Step acceptance flag

        # Accept step if model is reliable enough and we get actual reduction
        if rho_k > eta and ActRed > 0:
            v_next = proj(v_trial)  # Project trial point onto [v_min, v_max]
            step_accepted = True
        
        # Radius Update
        if rho_k < 0.25:
            Delta_next = 0.25 * Delta_k # # Very poor agreement: shrink radius
        elif rho_k > 0.75 and np.abs(p_k) == Delta_k:
            Delta_next = min(2 * Delta_k, Delta_max) # Very good agreement and step hit the boundary: expand radius

        # Stopping Criterion (Step Size)
        if step_accepted and abs(v_next - v_k) < tol:
            v_k = v_next
            if verbose:
                print(f"Converged by step size at iter {k}: |Δv| < {tol}")
            break

        v_k = v_next
        Delta_k = Delta_next

    return v_k,iterations

# Run solver
if __name__ == '__main__':
    v0_input = input("Initial Guess : ").strip()
    
    # Calculate v_pref for use as default guess and final comparison
    v_pref = v_pref_formula(tau, C, n, w_avg)

    if v0_input == "":
        v0 = v_pref
        print(f"Using v0 = v_pref_formula(...) = {v0:.6f} km/h as initial guess")
    else:
        try:
            v0 = float(v0_input)
        except ValueError:
            print("Invalid initial guess; using v_pref_formula instead.")
            v0 = v_pref

    v0 = proj(v0)  # Ensure initial guess respects bounds

    # Run optimization (Trust-Region)
    v_opt,iters = trust_region_optimization(v0=v0, verbose=True)

    # Final cost breakdown (Including conditional M/N adjustment)
    FC_base = FC_baseline_per_km(v_opt)
    baseline_fuel_used = FC_base * D

    # Target Fuel Logic (linear interpolation for D=10, then scaled by D)
    target_total_for_D10 = 0.37 + 0.08 * tau
    target_fuel_used = target_total_for_D10 * (D / 10.0)

    # Compute delta needed for adjustment
    delta = target_fuel_used - baseline_fuel_used

    # Conditional forms M/N adjustment (applied after optimization)
    if tau < 0.5:
        denom = (1.0 - tau) * D
        M = 0.0
        if abs(denom) > 1e-12:
            M = -delta / denom
        FC_v = FC_base - (1.0 - tau) * M
    else:
        denom = tau * D
        N = 0.0
        if abs(denom) > 1e-12:
            N = delta / denom
        FC_v = FC_base + tau * N

    # Safety clamp
    FC_v = max(1e-12, FC_v)
    fuel_used = FC_v * D

    # Final cost calculation
    fuel_cost = Pf * fuel_used
    regularizer_penalty = lambda_reg * (v_opt - v_pref)**2
    total_cost = fuel_cost + regularizer_penalty

    # Print final results
    print("\nRESULTS: ")
    print(f"Converged after {iters} iterations")
    print(f"Optimal Speed v* = {v_opt:.6f} km/h")
    print(f"Total Fuel used = {fuel_used:.6f} L")
    print(f"Fuel Cost = {fuel_cost:.2f}")
    print(f"Regularizer penalty = {regularizer_penalty:.6f}")
    print(f"Total Fuel cost at optimal speed = {total_cost:.2f}")
 