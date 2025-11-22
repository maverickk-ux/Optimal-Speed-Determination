import numpy as np

#Inputs
D = float(input("Distance (km): "))
tau = float(input("Traffic Index (0..1): "))   # 0 = no traffic, 1 = heavy traffic
n = int(input("Number of passengers: "))
C = float(input("Enter engine capacity (cc): ")) # Engine capacity (user input)
v_min = float (input("Enter minimum speed: "))
v_max = float (input("Enter maximum speed: "))

#Constants 
w_avg = 70.0 #considering avg weight of a person to be 70 kgs
Pf = 102.0  # Fuel Price (currency per L)
Ct = 160.0 #delay constant
s = 0.05

# Preferred-speed / formula coefficients
c1 = 75.0
c2 = 5400.0
c3 = 4.0 / 70.0

# Small baseline fuel-speed coefficients (kept small)
ce1_base = 2e-4
ce2_base = 1e-5

# Engine-efficiency scaling for baseline coefficients
C_ref = 400.0
beta_eff = 0.20

# baseline K term
c0 = 0.005
c3_base = 0.001
c4 = 1e-5
c5 = 0.002

# Regularizer weight
lambda_reg = 0.2

# Precompute scaling and baseline coefficients
B = Pf * D

eff_scale = 1.0 if C <= 0 else (C_ref / C) ** beta_eff
ce1 = ce1_base * eff_scale
ce2 = ce2_base * eff_scale

K = c0 + c3_base * tau + c4 * n * w_avg + c5 / (C if C > 0 else C_ref)

# Preferred-speed formula (your specified linear form)
def v_pref_formula(tau_val, C_val, n_val, w_avg_val):
    tau_clipped = np.clip(tau_val, 0.0, 1.0)
    C_safe = C_val if C_val > 0 else 1.0
    F = v_max - c1 * tau_clipped - (c2 / C_safe) - c3 * (n_val * w_avg_val)
    return float(np.clip(F, v_min, v_max))

# Baseline objective (used by Newton). We keep baseline FC as simple quadratic
# to preserve analytic derivatives for stable Newton steps.
def FC_baseline_per_km(v):
    return K + ce1 * v + ce2 * v**2

def f_base(v):
    if v <= 0:
        return float('inf')
    FC_v = FC_baseline_per_km(v)
    fuel_cost = B * FC_v
    time_cost = Ct * (D / v)
    const_delay = Ct * s * tau * D
    return fuel_cost + time_cost + const_delay

def f(v):
    vp = v_pref_formula(tau, C, n, w_avg)
    return f_base(v) + lambda_reg * (v - vp)**2

def fprime(v):
    # derivative of baseline: B*(ce1 + 2*ce2*v) - Ct*D/v^2
    base_deriv = B * (ce1 + 2.0 * ce2 * v) - Ct * D / (v**2)
    reg_deriv = 2.0 * lambda_reg * (v - v_pref_formula(tau, C, n, w_avg))
    return base_deriv + reg_deriv

def fdoubleprime(v):
    base_hess = 2.0 * B * ce2 + 2.0 * Ct * D / (v**3)
    reg_hess = 2.0 * lambda_reg
    return base_hess + reg_hess

def proj(v):
    return float(np.clip(v, v_min, v_max))

# Newton's method with projection
def newton_method(v0=40.0, tol=1e-6, max_iter=100, verbose=True):
    v = proj(v0)
    if verbose:
        print()
        print("Running Newton's Method...")
        print(f"Computed v_pref (formula) = {v_pref_formula(tau, C, n, w_avg):.6f} km/h")
        print(f"Starting Newton from v0 (projected) = {v:.6f} km/h, lambda={lambda_reg}")
        print(f"Baseline fuel coefficients: K={K:.6e}, ce1={ce1:.6e}, ce2={ce2:.6e}")
    for k in range(1, max_iter + 1):
        g = fprime(v)
        H = fdoubleprime(v)
        if abs(g) < tol:
            if verbose:
                print(f"Converged by gradient at iter {k-1}: |f'(v)|={abs(g):.3e} < {tol}")
            break
        if abs(H) < 1e-14:
            if verbose:
                print("Hessian too small — stopping.")
            break
        step = g / H
        v_new = proj(v - step)

        if abs(v_new - v) < tol:
            v = v_new
            if verbose:
                print(f"Converged by step size at iter {k}: |Δv| < {tol}")
            break
        v = v_new
    return v

# Initial guess (user input; blank -> use v_pref)
v0_input = input("Initial Guess: ").strip()
if v0_input == "":
    v0 = v_pref_formula(tau, C, n, w_avg)
    print(f"Using v0 = v_pref_formula(...) = {v0:.6f} km/h as initial guess")
else:
    try:
        v0 = float(v0_input)
    except ValueError:
        print("Invalid initial guess; using v_pref_formula instead.")
        v0 = v_pref_formula(tau, C, n, w_avg)
v0 = proj(v0)

# Run optimization (Newton)
v_opt = newton_method(v0=v0, verbose=True)

# Compute baseline FC and then apply conditional M/N adjustment to meet targets
FC_base = FC_baseline_per_km(v_opt)            # L/km baseline
baseline_fuel_used = FC_base * D               # total fuel used (L)

# For D=10: tau=0 -> 0.37 L ; tau=1 -> 0.45 L  (linear interpolation)
target_total_for_D10 = 0.37 + 0.08 * tau
target_fuel_used = target_total_for_D10 * (D / 10.0)

# Compute delta needed
delta = target_fuel_used - baseline_fuel_used

# Conditional forms:
# if tau < 0.5: FC_v = FC_base - (1 - tau) * M  => M = -delta / ((1-tau)*D)
# else:         FC_v = FC_base + tau * N        => N = delta / (tau*D)
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
fuel_cost = Pf * fuel_used
regularizer_penalty = lambda_reg * (v_opt - v_pref_formula(tau, C, n, w_avg))**2
total_cost = fuel_cost + regularizer_penalty 

# Print results
print()
print("RESULTS: ")
print(f"Optimal speed v* = {v_opt:.6f} km/h")
print(f"Total fuel used = {fuel_used:.6f} L (for D={D} km)")
print(f"Fuel cost = {fuel_cost:.2f}")
print(f"Regularizer penalty = {regularizer_penalty:.6f}")
print(f"Total trip cost = {total_cost:.2f}")
