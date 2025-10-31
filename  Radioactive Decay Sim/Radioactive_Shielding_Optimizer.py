import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
from itertools import product

# ---------------------------
# Utility / physics functions
# ---------------------------

def linear_mu_from_mass(mu_mass, density):
    """
    Convert mass attenuation coefficient (cm^2/g) to linear attenuation (cm^-1):
      mu_lin = mu_mass * density (g/cm^3)
    """
    return mu_mass * density

def transmission_exponential(mu_lin, thickness):
    """
    Exponential attenuation: I/I0 = exp(-mu_lin * thickness)
    mu_lin: linear attenuation coefficient (1/cm)
    thickness: cm
    """
    return np.exp(-mu_lin * thickness)

def required_thickness_for_target(mu_lin, target_fraction):
    """
    Solve exp(-mu * x) = target_fraction -> x = -ln(target)/mu
    """
    if target_fraction <= 0:
        raise ValueError("target_fraction must be > 0")
    return -np.log(target_fraction) / mu_lin

# ---------------------------
# Monte Carlo validator
# ---------------------------

def mc_transmission(mu_lin, thickness, N=100000, seed=None):
    """
    Monte Carlo straight-line photon sampling:
    - sample path length s from exponential distribution with mean 1/mu_lin
    - transmission fraction = fraction with s > thickness
    This emulates photons travelling perpendicular to slab (no scattering).
    """
    if seed is not None:
        np.random.seed(seed)
    # sample s = -ln(U)/mu_lin
    U = np.random.random(N)
    s = -np.log(U) / mu_lin
    transmitted = np.sum(s > thickness) / N
    se = np.sqrt(transmitted * (1 - transmitted) / N)
    return transmitted, se

# ---------------------------
# Simple single-material optimizer
# ---------------------------

def single_material_solution(material, target_fraction):
    """
    material: dict with keys 'name', 'density' (g/cm3), either 'mu_mass' (cm^2/g) or 'mu_lin' (1/cm)
    target_fraction: allowed fraction of transmitted intensity (I/I0)
    Returns: (required_thickness_cm, areal_density_g_per_cm2, expected_transmission)
    """
    if 'mu_lin' in material:
        mu_lin = material['mu_lin']
    elif 'mu_mass' in material and 'density' in material:
        mu_lin = linear_mu_from_mass(material['mu_mass'], material['density'])
    else:
        raise ValueError("Material must include 'mu_lin' or ('mu_mass' and 'density')")
    thickness = required_thickness_for_target(mu_lin, target_fraction)
    areal_density = material.get('density', 1.0) * thickness  # g/cm^2
    return thickness, areal_density, transmission_exponential(mu_lin, thickness)

# ---------------------------
# Two-material brute-force optimizer
# ---------------------------

def optimize_two_materials(mat1, mat2, target_fraction,
                           max_thickness1=50.0, max_thickness2=50.0,
                           steps1=201, steps2=201):
    """
    Grid search over thicknesses of mat1 and mat2 to find combination that:
      - achieves transmission <= target_fraction
      - minimizes total areal density (mass per unit area)
    matX: material dict as in single_material_solution
    max_thicknessX: search limits in cm
    stepsX: grid resolution
    Returns: dict with best solution and full search arrays (optional)
    """
    # get mu_lin and density
    def mu_lin_from(mat):
        if 'mu_lin' in mat:
            return mat['mu_lin']
        return linear_mu_from_mass(mat['mu_mass'], mat['density'])
    mu1 = mu_lin_from(mat1)
    mu2 = mu_lin_from(mat2)
    rho1 = mat1.get('density', 1.0)
    rho2 = mat2.get('density', 1.0)

    t1_vals = np.linspace(0, max_thickness1, steps1)
    t2_vals = np.linspace(0, max_thickness2, steps2)

    best = None
    best_details = None

    # grid search
    for t1 in t1_vals:
        # Transmission after layer 1 then layer 2 (assume same beam normal incidence)
        trans1 = np.exp(-mu1 * t1)
        # For vectorization: compute trans2 for all t2
        trans2 = np.exp(-mu2 * t2_vals)
        total_trans = trans1 * trans2  # product of attenuations
        mask_ok = total_trans <= target_fraction
        if not np.any(mask_ok):
            continue
        # compute areal density
        areal = rho1 * t1 + rho2 * t2_vals
        areal_ok = areal[mask_ok]
        t2_ok = t2_vals[mask_ok]
        trans_ok = total_trans[mask_ok]
        # find minimal areal among ok combos for this t1
        idx = np.argmin(areal_ok)
        this_best_areal = areal_ok[idx]
        this_best_t2 = t2_ok[idx]
        this_best_trans = trans_ok[idx]
        if (best is None) or (this_best_areal < best):
            best = this_best_areal
            best_details = {
                't1_cm': t1,
                't2_cm': this_best_t2,
                'areal_density_g_per_cm2': this_best_areal,
                'transmission': this_best_trans
            }

    return best_details

# ---------------------------
# Example materials (illustrative values)
# Replace these with real coefficients for your radiation type/energy.
# Units:
#  - mu_mass in cm^2/g (mass attenuation)
#  - density in g/cm3
#  - mu_lin in 1/cm if provided directly
# ---------------------------

materials = {
    'lead':     {'name': 'Lead',     'density': 11.34, 'mu_mass': 0.044},  # example mass coeff (1 MeV gamma-ish)
    'concrete': {'name': 'Concrete', 'density': 2.3,  'mu_mass': 0.035},  # illustrative
    'water':    {'name': 'Water',    'density': 1.0,  'mu_mass': 0.034},
    # you can add 'mu_lin' directly if you have it:
    # 'steel': {'name': 'Steel','mu_lin': 0.12, 'density': 7.8}
}

# Precompute mu_lin for convenience
for m in materials.values():
    if 'mu_lin' not in m:
        m['mu_lin'] = linear_mu_from_mass(m['mu_mass'], m['density'])

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Problem statement:
    # Suppose you have a gamma source and want to reduce intensity by a factor of 1e6 (I/I0 <= 1e-6)
    target = 1e-6

    print("TARGET transmission (I/I0) =", target)
    print("\nSingle-material results (illustrative coefficients):")
    for key, mat in materials.items():
        thickness, areal, trans = single_material_solution(mat, target)
        print(f"  {mat['name']:8s}: thickness = {thickness:.2f} cm, areal density = {areal:.2f} g/cm^2")

    # Plot Transmission vs thickness for each material
    thickness_range = np.linspace(0, 50, 501)  # cm
    plt.figure(figsize=(8,5))
    for key, mat in materials.items():
        mu = mat['mu_lin']
        T = np.exp(-mu * thickness_range)
        plt.semilogy(thickness_range, T, label=f"{mat['name']}")
    plt.axhline(target, color='k', linestyle='--', label=f"target {target:.0e}")
    plt.xlabel("Thickness (cm)")
    plt.ylabel("Transmission I/I0 (log scale)")
    plt.title("Transmission vs Thickness (illustrative materials)")
    plt.legend()
    plt.grid(True, which='both', ls=':', alpha=0.6)
    plt.show()

    # Two-material optimization example: minimize total mass-per-area while meeting target
    print("\nTwo-material optimization (minimize areal density to meet target):")
    matA = materials['concrete']
    matB = materials['lead']
    # search up to 100 cm of concrete and 20 cm of lead
    best = optimize_two_materials(matA, matB, target_fraction=target,
                                  max_thickness1=100.0, max_thickness2=20.0,
                                  steps1=401, steps2=401)
    if best:
        print(f"  Best combo: {matA['name']} {best['t1_cm']:.2f} cm + {matB['name']} {best['t2_cm']:.2f} cm")
        print(f"  -> Areal density = {best['areal_density_g_per_cm2']:.2f} g/cm^2, Transmission = {best['transmission']:.2e}")
    else:
        print("  No combination in the search range met the target. Increase max thickness or adjust materials.")

    # Monte Carlo validation for single-material example
    mat = materials['lead']
    thickness_example = 10.0  # cm
    trans_analytic = transmission_exponential(mat['mu_lin'], thickness_example)
    trans_mc, se = mc_transmission(mat['mu_lin'], thickness_example, N=200000, seed=42)
    print(f"\nMonte Carlo check (straight-line, no scattering) for {mat['name']} {thickness_example:.1f} cm:")
    print(f"  Analytic transmission = {trans_analytic:.3e}")
    print(f"  MC transmission = {trans_mc:.3e} ± {1.96*se:.3e} (95% CI)")

# End of script