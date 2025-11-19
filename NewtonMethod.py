def FC(v):
    return c0 + c1*v + c2*(v**2) + c3*t + c4*(n*wavg) + c5/C

def f(v):
    if v <= 0:
        return float('inf')
    fuel_cost = Pf * D * FC(v)
    time_cost = Ct * (D/v + s * t * D)
    return fuel_cost + time_cost

def fprime(v):
    B = Pf * D
    return B*(c1 + 2*c2*v) - Ct * D / (v**2)

def fdoubleprime(v):
    B = Pf * D
    return 2*B*c2 + 2*Ct*D / (v**3)

#Taking inputs as per our objective function parameters
D = float(input("Trip distance (D): "))
t = float(input("Traffic index (tau 0-1): "))
n = int(input("Number of people (n): "))
wavg = float(input("Average weight per person (wavg): "))
C = float(input("Engine factor (C): "))

Pf = float(input("Fuel price (Pf): "))
Ct = float(input("Cost of time per hour (Ct): "))
s = float(input("Traffic delay factor (s): "))

# Fuel model coefficients
c0 = 0.005
c1 = 2*(1e-4)
c2 = 1*(1e-5)
c3 = 0.001
c4 = 1*(1e-5)
c5 = 0.001

v_min = float(input("Minimum speed v_min: "))
v_max = float(input("Maximum speed v_max: "))

v = float(input("Initial guess for speed v0: "))

# ---- NEWTON'S METHOD ----
tol = 1e-6
max_iters = 100

for i in range(max_iters):
    #Finding first and second derivative of objective function
    g = fprime(v) #gradient of the fuction
    h = fdoubleprime(v) #hessian of the function

    if abs(h) < 1e-8:
        print("Hessian is too small. Stopping...")
        break

    newv = v - g/h

    # Keep inside feasible range
    if newv < v_min: newv = v_min
    if newv > v_max: newv = v_max

    if abs(newv - v) < tol:
        v = newv
        print(f"Iters: {i}")
        break

    v = newv

# ---- RESULTS ----
FC_v = FC(v)
fuel_used = D * FC_v
fuel_cost = Pf * fuel_used
time_cost = Ct * (D/v + s * t * D)
total_cost = fuel_cost + time_cost

print("\n--- RESULTS ---")
print("Optimal speed v* =", v, "km/h")
print("Fuel used =", fuel_used, "L")
print("Fuel cost =", fuel_cost)
print("Time cost =", time_cost)
print("Total trip cost =", total_cost)
