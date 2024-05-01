import numpy as np
import matplotlib.pyplot as plt
#constants
nb_iterations = 10000
lattice_size = 5
initial_state = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
magnetic_field = -0.8 #B
interaction_energy = 1.0 #J
inverse_temperature = 0.10 #beta
magnetic_moment = 1.0 #mu
burn_in_iterations = 1000
#sampling functions
def energy(state):
    return -magnetic_moment*magnetic_field*np.sum(state) - interaction_energy*np.sum([state[(i+1)%lattice_size, j]*state[i, j] + state[i, (j+1)%lattice_size]*state[i, j] + state[(i-1)%lattice_size, j]*state[i, j] + state[i, (j-1)%lattice_size]*state[i, j] for i in range(lattice_size) for j in range(lattice_size)])
def energyDelta(new_state,i,j):
    sigma = new_state[i,j]
    neighbour_contrib = new_state[(i+1)%lattice_size, j] + new_state[i, (j+1)%lattice_size] + new_state[(i-1)%lattice_size, j] + new_state[i, (j-1)%lattice_size]
    return -2*sigma*(magnetic_moment*magnetic_field + interaction_energy*np.sum(neighbour_contrib))
def magnetization(state):
    return magnetic_moment*np.sum(state)

def metropolis_hastings(state):
    energies = np.zeros(nb_iterations-burn_in_iterations)  
    magnetizations = np.zeros(nb_iterations-burn_in_iterations)
    for iteration in range(nb_iterations):
        i, j = np.random.randint(lattice_size, size=2)
        delta_energy = energyDelta(state,i,j)
        if delta_energy < 0:
            state[i, j] *= -1
        elif np.random.rand() < np.exp(-inverse_temperature*delta_energy):
            state[i, j] *= -1
        if iteration >= burn_in_iterations:
            energies[iteration-burn_in_iterations] = energy(state)
            magnetizations[iteration-burn_in_iterations] = magnetization(state)
    return energies, magnetizations
def heat_capacity(energies):
    return np.var(energies)*(inverse_temperature**2)/energies.size
def susceptibility(magnetizations):
    return np.var(magnetizations)*(inverse_temperature)/magnetizations.size
#main
energies, magnetizations = metropolis_hastings(initial_state)
#find and plot heat capacity and suceptibility for different temperatures, in this case, for different values of beta, and different values of magnetic field and interaction energy
inverse_temperatures = np.linspace(0.01, 1.0, 100)
heat_capacities = np.zeros(inverse_temperatures.size)
susceptibilities = np.zeros(inverse_temperatures.size)
for i, inverse_temperature in enumerate(inverse_temperatures):
    energies, magnetizations = metropolis_hastings(initial_state)
    heat_capacities[i] = heat_capacity(energies)
    susceptibilities[i] = susceptibility(magnetizations)
plt.plot(inverse_temperatures, heat_capacities)
plt.xlabel('Inverse temperature')
plt.ylabel('Heat capacity')
plt.show()
plt.plot(inverse_temperatures, susceptibilities)
plt.xlabel('Inverse temperature')
plt.ylabel('Susceptibility')
plt.show()
