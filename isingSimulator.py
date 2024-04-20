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
# Create a figure and a 3x1 subplot (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1)

    # Plot histograms
    axs[0].hist(samples[:, 0], bins=30)
    axs[0].set_title('X coordinates')

    axs[1].hist(samples[:, 1], bins=30)
    axs[1].set_title('Y coordinates')

    axs[2].hist(samples[:, 2], bins=30)
    axs[2].set_title('Z coordinates')

    # Display the plot
    plt.tight_layout()
    plt.show()
def plot_vs_samples():
    fig, axs = plt.subplots(3, 1)
    axs[0].scatter(samples[:, 0], samples[:, 1], label='s1 vs s2')
    axs[0].legend()
    axs[1].scatter(samples[:, 0], samples[:, 2], label='s1 vs s3')
    axs[1].legend()
    axs[2].scatter(samples[:, 1], samples[:, 2], label='s2 vs s3')
    axs[2].legend()    
    plt.legend()
    plt.tight_layout()
    plt.show()
#main
if __name__ == '__main__':
    metropolis_hastings()
    plot_samples()
    plot_vs_samples()