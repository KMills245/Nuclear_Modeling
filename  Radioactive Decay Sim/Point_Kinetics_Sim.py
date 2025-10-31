import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def point_kinetics(y, t, rho, beta_i, lambda_i, Lambda, S):
    """
    y[0] = neutron density n
    y[1:] = precursor concentrations C_i
    """
    n = y[0]
    C = y[1:]
    
    beta = np.sum(beta_i)
    dn_dt = ((rho - beta) / Lambda) * n + np.sum(lambda_i * C) + S
    dC_dt = beta_i / Lambda * n - lambda_i * C
    
    return np.concatenate(([dn_dt], dC_dt))

# --- Parameters ---
Lambda = 1e-5              # prompt neutron generation time (s)
beta_i = np.array([0.00025, 0.0012, 0.0011, 0.0027, 0.0008, 0.00025])  # delayed neutron fractions
lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])        # precursor decay constants (1/s)
rho = 0.002                # reactivity (dimensionless)
S = 0                      # external neutron source

# Initial conditions
n0 = 1.0                   # initial neutron density (arbitrary units)
C0 = beta_i / (lambda_i * Lambda) * n0  # equilibrium precursor concentrations
y0 = np.concatenate(([n0], C0))

# Time grid
t = np.linspace(0, 10, 1000)

# Solve the system
solution = odeint(point_kinetics, y0, t, args=(rho, beta_i, lambda_i, Lambda, S))
n = solution[:, 0]
C = solution[:, 1:]

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.plot(t, n, label='Neutron Density (n)')
for i in range(len(beta_i)):
    plt.plot(t, C[:, i], '--', label=f'Precursor Group {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Concentration / Neutron Density')
plt.title('Point Kinetics with Delayed Neutron Precursors')
plt.legend()
plt.grid(True)
plt.show()