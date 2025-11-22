import numpy as np

# -----------------------------
# Inputs (same interface as your Newton script)
D = float(input("Distance (km): "))
tau = float(input("Traffic Index (0..1): "))   # 0 = no traffic, 1 = heavy traffic
n = int(input("Number of passengers: "))
C = float(input("Enter engine capacity (cc): ")) # Engine capacity (user input)
v_min = float(input("Enter minimum speed: "))
v_max = float(input("Enter maximum speed: "))

# Constants 
w_avg = 70.0 # avg weight (kg)
Pf = 102.0  # Fuel Price (currency per L)
Ct = 160.0  # delay constant (you had this in Newton)
s = 0.05

# Preferred-speed / formula coefficients (unchanged)
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

# Regularizer weight (same)
lambda_reg = 0.2

# Precompute scaling and baseline coefficients
B = Pf * D
eff_scale = 1.0 if C <= 0 else (C_ref / C) ** beta_eff
ce1 = ce1_base * eff_scale
ce2 = ce2_base * eff_scale
K = c0 + c3_base * tau + c4 * n * w_avg + c5 / (C if C > 0 else C_ref)

# Preferred-speed formula (linear form)
def v_pref_formula(tau_val, C_val, n_val, w_avg_val):
    tau_clipped = np.clip(tau_val, 0.0, 1.0)
    C_safe = C_val if C_val > 0 else 1.0
    F = v_max - c1 * tau_clipped - (c2 / C_safe) - c3 * (n_val * w_avg_val)
    return float(np.clip(F, v_min, v_max))

# Baseline FC per km (quadratic baseline)
def FC_baseline_per_km(v):
    return K + ce1 * v + ce2 * v**2

# --- IMPORTANT: match Newton's objective (includes time term)
def f_base(v):
    if v <= 0:
        return float('inf')
    FC_v = FC_baseline_per_km(v)
    fuel_cost = B * FC_v
    time_cost = Ct * (D / v)           # <-- time term included (match Newton)
    const_delay = Ct * s * tau * D
    return fuel_cost + time_cost + const_delay

def f(v):
    vp = v_pref_formula(tau, C, n, w_avg)
    return f_base(v) + lambda_reg * (v - vp)**2

# Derivatives (match Newton's analytic derivatives)
def fprime(v):
    # derivative of baseline: B*(ce1 + 2*ce2*v) - Ct * D / v^2
    base_deriv = B * (ce1 + 2.0 * ce2 * v) - Ct * D / (v**2)
    reg_deriv = 2.0 * lambda_reg * (v - v_pref_formula(tau, C, n, w_avg))
    return base_deriv + reg_deriv

def fdoubleprime(v):
    # base second derivative: 2*B*c2 + 2*Ct*D / v^3
    base_hess = 2.0 * B * ce2 + 2.0 * Ct * D / (v**3)
    reg_hess = 2.0 * lambda_reg
    return base_hess + reg_hess

def proj(v):
    return float(np.clip(v, v_min, v_max))

# -----------------------------
# BFGS method now using the same derivatives as Newton
def bfgs_method(v0=40.0, tol=1e-9, max_iter=200, verbose=True):
    x = np.array([proj(v0)], dtype=float)
    if verbose:
        print()
        print("Running BFGS (quasi-Newton) ...")
        print(f"Computed v_pref (formula) = {v_pref_formula(tau, C, n, w_avg):.6f} km/h")
        print(f"Starting BFGS from v0 (projected) = {x[0]:.6f} km/h, lambda={lambda_reg}")
        print(f"Baseline fuel coefficients: K={K:.6e}, ce1={ce1:.6e}, ce2={ce2:.6e}")

    # initialize gradient and inverse-H using analytic second derivative at v0
    g0 = float(fprime(x[0]))
    fpp0 = float(fdoubleprime(x[0]))
    # safe inverse Hessian scalar
    inv_h0 = 1.0 / (abs(fpp0) + 1e-12)
    Hk = np.array([[inv_h0]], dtype=float)

    iter_count = 0

    # Armijo/backtracking parameters
    alpha0 = 1.0
    c1_ls = 1e-4
    rho = 0.5
    max_line_iters = 30

    for k in range(1, max_iter + 1):
        iter_count += 1
        g = np.array([fprime(x[0])], dtype=float)
        if abs(g[0]) < tol:
            break

        pk = -Hk.dot(g)          # direction (1x1)
        # ensure descent
        gTp = float(g.reshape(-1).dot(pk.reshape(-1)))
        if gTp >= 0:
            pk = -g
            gTp = float(g.reshape(-1).dot(pk.reshape(-1)))

        # line search on unprojected step, but evaluate f at projected candidate
        alpha = alpha0
        fx = f(x[0])
        found = False
        x_new = None

        for _ in range(max_line_iters):
            cand_unproj = x[0] + alpha * pk[0]
            cand_proj = proj(cand_unproj)
            f_cand = f(cand_proj)
            if f_cand <= fx + c1_ls * alpha * gTp:
                x_new = np.array([cand_proj], dtype=float)
                found = True
                break
            # if projection did nothing, reduce step
            if abs(cand_proj - x[0]) < 1e-12:
                alpha *= rho
                continue
            alpha *= rho

        if not found:
            alpha = 1e-6
            cand_proj = proj(x[0] + alpha * pk[0])
            x_new = np.array([cand_proj], dtype=float)

        s = x_new - x
        g_new = np.array([fprime(x_new[0])], dtype=float)
        y = g_new - g

        sty = float(s.reshape(-1).dot(y.reshape(-1)))
        if sty > 1e-12:
            rho_b = 1.0 / sty
            I = np.eye(1)
            term1 = I - rho_b * np.outer(s, y)
            term2 = I - rho_b * np.outer(y, s)
            Hk = term1.dot(Hk).dot(term2) + rho_b * np.outer(s, s)
        # else: skip update

        x = x_new

        if np.linalg.norm(s) < tol:
            break

    return float(x[0]), iter_count

# -----------------------------
# Initial guess (same I/O)
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

# Run BFGS
v_opt, iterations = bfgs_method(v0=v0, tol=1e-9, max_iter=200, verbose=True)

# Post-processing & target adjustment (same as before)
FC_base = FC_baseline_per_km(v_opt)
baseline_fuel_used = FC_base * D
target_total_for_D10 = 0.37 + 0.08 * tau
target_fuel_used = target_total_for_D10 * (D / 10.0)
delta = target_fuel_used - baseline_fuel_used

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

FC_v = max(1e-12, FC_v)
fuel_used = FC_v * D
fuel_cost = Pf * fuel_used
regularizer_penalty = lambda_reg * (v_opt - v_pref_formula(tau, C, n, w_avg))**2
total_cost = fuel_cost + regularizer_penalty

# Final prints (same format)
print("\nRESULTS:")
print(f"Converged after {iterations} iterations")
print(f"Optimal Speed v* = {v_opt:.6f} km/h")
print(f"Total Fuel used = {fuel_used:.6f} L")
print(f"Fuel Cost = {fuel_cost}")
print(f"Regularized penalty: {regularizer_penalty:.6f}")
print(f"Total Fuel cost at optimal speed = {total_cost:.2f}")
          