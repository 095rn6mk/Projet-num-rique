import numpy as np
import matplotlib.pyplot as plt

#constantes
nb_iterations = 2000
variance_proposition = 0.01
etat_initial = np.array([0, 0, 0])
matrice_transition = np.array([[0.065, 0.036, -0.056], [0.036, 0.182, -0.078], [-0.056, -0.078, 0.110]])
echantillons = np.zeros((nb_iterations, 3))
inverse_matrice = np.linalg.inv(matrice_transition)
#fonctions d'échantillonage
def proposition_stationnaire():
    return np.random.multivariate_normal(np.zeros(3), variance_proposition*np.eye(3))

def proposition(etat):
    return etat + proposition_stationnaire()

def distribution_cible(etat):
    return np.exp(-etat.dot(inverse_matrice).dot(etat)/2)

def probabilite_acceptation(etat, nouvel_etat):
    return min(1, distribution_cible(nouvel_etat)/distribution_cible(etat))

#metropolis hastings
def metropolis_hastings():
    etat = etat_initial
    for i in range(nb_iterations):
        nouvel_etat = proposition_stationnaire()
        if np.random.rand() < probabilite_acceptation(etat, nouvel_etat):
            etat = nouvel_etat
        echantillons[i] = etat

#fonctions de tracé
def tracer_echantillons(comparaison=False):
    fig, axs = plt.subplots(3, 1)
    axs[0].hist(2*echantillons[:, 0], bins=30)
    axs[0].set_title('Coordonnées X')
    axs[1].hist(2*echantillons[:, 1], bins=30)
    axs[1].set_title('Coordonnées Y')
    axs[2].hist(2*echantillons[:, 2], bins=30)
    axs[2].set_title('Coordonnées Z')
    if comparaison:
        x = np.linspace(-1, 1, 100)
        y = np.exp(-x**2/(2*matrice_transition[0, 0]))/np.sqrt(2*np.pi*matrice_transition[0, 0])
        axs[0].plot(x, y*echantillons.size*2/30, color='red')
        x= np.linspace(-1, 1, 100)
        y = np.exp(-x**2/(2*matrice_transition[1, 1]))/np.sqrt(2*np.pi*matrice_transition[1, 1])
        axs[1].plot(x, y*echantillons.size*2/30, color='green')
        x= np.linspace(-1, 1, 100)
        y = np.exp(-x**2/(2*matrice_transition[2, 2]))/np.sqrt(2*np.pi*matrice_transition[2, 2])
        axs[2].plot(x, y*echantillons.size*2/30, color='blue')
    plt.tight_layout()
    plt.show()

def tracer_vs_echantillons(comparaison=False):
    fig, axs = plt.subplots(3, 1)
    axs[0].scatter(echantillons[:, 0], echantillons[:, 1], label='s1 vs s2')
    axs[0].legend()
    axs[1].scatter(echantillons[:, 0], echantillons[:, 2], label='s1 vs s3')
    axs[1].legend()
    axs[2].scatter(echantillons[:, 1], echantillons[:, 2], label='s2 vs s3')
    axs[2].legend()    
    plt.legend()
    plt.tight_layout()
    plt.show()

def tracer_3D(comparaison=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(echantillons[:, 0], echantillons[:, 1], echantillons[:, 2])
    plt.show()

#main
if __name__ == '__main__':
    metropolis_hastings()
    tracer_echantillons()
    tracer_vs_echantillons()
    tracer_3D()
    tracer_echantillons(True)