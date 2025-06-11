import numpy as np 

def slab_transmission(Sig_t, thickness, N, isotropic = false): 
    """Compute the fraction of neutrons that leak through a slab
    Inputs: 
    Sig_t: THe total macroscopic x-section
    Thickness: Width of the slab 
    N: Number of neutrons to simulate 
    Isotropic: Are the neutrons isotropuc or a beam 

    Returns: 
    Transmission: The fraction of neutrons that made it through 
    """

    if(isotropic):
        mu = np.random.random(N)
    else:
        mu = np.ones(N)
    thetas = np.random.random(N)
    x = -np.log(1 - thetas) / Sig_t 
    transmisison = np.sum(x > thickness / mu) / N

    # For a small number of neutrons we'll ouput a little more 
    if (N <= 1000): 
        plt.scatter(x * mu, np.arrange(N))
        plt.xlabel("Distance traveled into slab") 
        plt.ylabel("Neutron Number")
return transmission 

### Test the function with a small number of neutrons

Sigma_t = 2.0
thickness = 3.0
N = 1000


transmission = slab_transmission(Sigma_t, thickness, N, isotropic = True) 
print("Out of", N,"neutrons only", int(transmission * N), "made it through.\n The fraction that made it through was", transmission) 