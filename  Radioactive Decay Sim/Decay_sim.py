import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt  # type: ignore

def fission_prod_chain(y, t, lambda_a, lambda_b):
    na, nb, nc = y
    dnadt = s_a + (-lambda_a * na)   # Decay of na to nb
    dnbdt = lambda_a * na - lambda_b * nb  # Conversion of na to nb and decay of nb
    dncdt = lambda_b * nb  # Decay of nb to nc
    return [dnadt, dnbdt, dncdt]

# Initial Parameters 
s_a = 0.5 # Constant Source Rate
na0 = 50  # Initial number of na
lambda_a = 0.10  # Decay constant for na

nb0 = 0     # Initial number of nb
lambda_b = 0.03  # Decay constant for nb

nc0 = 0     # Initial number of nc

# Time range (longer period)
t = np.linspace(0, 200, 1000)  # Longer time range for clearer visualization

# Initial conditions: [na, nb, nc]
initial_conditions = [na0, nb0, nc0]

# Solve the ODE
solution = odeint(fission_prod_chain, initial_conditions, t, args=(lambda_a, lambda_b))

# Extract solutions for each variable
na, nb, nc = solution.T

# Plot results
plt.plot(t, na, label='na (Initial Particles)')
plt.plot(t, nb, label='nb (Intermediate Particles)')
plt.plot(t, nc, label='nc (Final Particles)')
plt.title("Radioactive Decay Simulation")
plt.xlabel("Time")
plt.ylabel("Remaining Particles")
plt.legend()