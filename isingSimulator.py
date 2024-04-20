import numpy as np
import matplotlib.pyplot as plt

#constants
nb_iterations = 1000
proposition_variance = 0.01
initial_state = np.array([0, 0, 0])
transition_matrix = np.array([[0.065, 0.036, -0.056], [0.036, 0.182, -0.078], [-0.056, -0.078, 0.110]])
samples = np.zeros((nb_iterations, 3))
#sampling functions
def stationnary_proposal():
    return np.random.multivariate_normal(np.zeros(3), proposition_variance*np.eye(3))
def proposoal(state):
    return state + stationnary_proposal()
def target_distribution(state):
    return np.exp(-state.dot(transition_matrix).dot(state)/2)
def acceptance_probability(state, new_state):
    return min(1, target_distribution(new_state)/target_distribution(state))
#metropolis hastings algorithm
def metropolis_hastings():
    state = initial_state
    for i in range(nb_iterations):
        new_state = stationnary_proposal()
        if np.random.rand() < acceptance_probability(state, new_state):
            state = new_state
        samples[i] = state
#plotting functions
def plot_samples():
    plt.plot(samples[:, 0], label='x')
    plt.plot(samples[:, 1], label='y')
    plt.plot(samples[:, 2], label='z')
    plt.legend()
    plt.show()
#main
metropolis_hastings()
plot_samples()