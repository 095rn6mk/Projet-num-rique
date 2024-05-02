import numpy as np
import matplotlib.pyplot as plt
#constants
nb_iterations = 2000
taille_grille = 5
etat_initial = np.random.choice([-1, 1], size=(taille_grille, taille_grille))
champ_magnetique = -0.8 #B
energie_interaction = 1.0 #J
temperature_inverse = 0.10 #beta
moment_magnetique = 1.0 #mu
iterations_avant_echantillonnage = 1000
#sampling functions
def energie(etat):
    return -moment_magnetique*champ_magnetique*np.sum(etat) - energie_interaction*np.sum([etat[(i+1)%taille_grille, j]*etat[i, j] + etat[i, (j+1)%taille_grille]*etat[i, j] + etat[(i-1)%taille_grille, j]*etat[i, j] + etat[i, (j-1)%taille_grille]*etat[i, j] for i in range(taille_grille) for j in range(taille_grille)])
def deltaEnergie(nouvel_etat,i,j):
    sigma = nouvel_etat[i,j]
    contrib_voisins = nouvel_etat[(i+1)%taille_grille, j] + nouvel_etat[i, (j+1)%taille_grille] + nouvel_etat[(i-1)%taille_grille, j] + nouvel_etat[i, (j-1)%taille_grille]
    return -2*sigma*(moment_magnetique*champ_magnetique + energie_interaction*np.sum(contrib_voisins))
def magnetisation(etat):
    return moment_magnetique*np.sum(etat)

def metropolis_hastings(etat):
    etat = etat.copy()
    energies = np.zeros(nb_iterations-iterations_avant_echantillonnage)  
    magnetisations = np.zeros(nb_iterations-iterations_avant_echantillonnage)
    for iteration in range(nb_iterations):
        i, j = np.random.randint(taille_grille, size=2)
        delta_energie = deltaEnergie(etat,i,j)
        if delta_energie < 0:
            etat[i, j] *= -1
        elif np.random.rand() < np.exp(-temperature_inverse*delta_energie):
            etat[i, j] *= -1
        if iteration >= iterations_avant_echantillonnage:
            energies[iteration-iterations_avant_echantillonnage] = energie(etat)
            magnetisations[iteration-iterations_avant_echantillonnage] = magnetisation(etat)
    return energies, magnetisations
def capacite_thermique(energies):
    return np.var(energies)*(temperature_inverse**2)/energies.size
def susceptibilite(magnetisations):
    return np.var(magnetisations)*(temperature_inverse)/magnetisations.size
#main
energies, magnetisations = metropolis_hastings(etat_initial)
#find and plot heat capacity and suceptibility for different temperatures, in this case, for different values of beta, and different values of magnetic field and interaction energy
temperatures_inverses = np.linspace(0.01, 1.0, 100)
capacites_thermiques = np.zeros(temperatures_inverses.size)
susceptibilites = np.zeros(temperatures_inverses.size)
for i, temperature_inverse in enumerate(temperatures_inverses):
    energies, magnetisations = metropolis_hastings(etat_initial)
    capacites_thermiques[i] = capacite_thermique(energies)
    susceptibilites[i] = susceptibilite(magnetisations)
plt.plot(temperatures_inverses, capacites_thermiques, color='IndianRed')
plt.xlabel('Temperature inverse')
plt.ylabel('Capacite thermique')
plt.show()
plt.plot(temperatures_inverses, susceptibilites, color='RoyalBlue')
plt.xlabel('Temperature inverse')
plt.ylabel('Susceptibilite')
plt.show()
#same thing for different values of magnetic field
#reset inverse temperature
temperature_inverse = 0.10
champs_magnetiques = np.linspace(-1.0, 1.0, 100)
capacites_thermiques = np.zeros(champs_magnetiques.size)
susceptibilites = np.zeros(champs_magnetiques.size)
for i, champ_magnetique in enumerate(champs_magnetiques):
    energies, magnetisations = metropolis_hastings(etat_initial)
    capacites_thermiques[i] = capacite_thermique(energies)
    susceptibilites[i] = susceptibilite(magnetisations)
plt.plot(champs_magnetiques, capacites_thermiques, color='IndianRed')
plt.xlabel('Champ magnetique')
plt.ylabel('Capacite thermique')
plt.show()
plt.plot(champs_magnetiques, susceptibilites, color='RoyalBlue')
plt.xlabel('Champ magnetique')
plt.ylabel('Susceptibilite')
plt.show()

#same thing for different values of interaction energy
#reset magnetic field
champ_magnetique = -0.8
energies_interaction = np.linspace(0.1, 2.0, 100)
capacites_thermiques = np.zeros(energies_interaction.size)
susceptibilites = np.zeros(energies_interaction.size)
for i, energie_interaction in enumerate(energies_interaction):
    energies, magnetisations = metropolis_hastings(etat_initial)
    capacites_thermiques[i] = capacite_thermique(energies)
    susceptibilites[i] = susceptibilite(magnetisations)
plt.plot(energies_interaction, capacites_thermiques, color='IndianRed')
plt.xlabel('Energie d\'interaction')
plt.ylabel('Capacite thermique')
plt.show()
plt.plot(energies_interaction, susceptibilites, color='RoyalBlue')
plt.xlabel('Energie d\'interaction')
plt.ylabel('Susceptibilite')
plt.show()