# scripts/analyze_results.py
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(positions, save_path="results/trajectories.png"):
    """
    Plot the trajectories of particles over time.

    Parameters:
        positions (list of ndarray): List of particle positions at each timestep.
        save_path (str): Path to save the trajectory plot.
    """
    num_particles = positions[0].shape[0]
    timesteps = len(positions)

    for i in range(num_particles):
        trajectory = np.array([positions[t][i] for t in range(timesteps)])
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Particle {i}")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Particle Trajectories")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Trajectory plot saved to {save_path}")

def compute_error(true_positions, predicted_positions):
    """
    Compute the mean squared error (MSE) between true and predicted positions.

    Parameters:
        true_positions (ndarray): Ground truth positions of particles.
        predicted_positions (ndarray): Predicted positions of particles.

    Returns:
        float: The mean squared error.
    """
    error = np.mean((true_positions - predicted_positions) ** 2)
    return error

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--positions_file", required=True, help="Path to positions file (numpy array)")
    parser.add_argument("--save_path", default="results/trajectories.png", help="Path to save trajectory plot")
    args = parser.parse_args()

    # Load particle positions
    positions = np.load(args.positions_file, allow_pickle=True)

    # Plot trajectories
    plot_trajectories(positions, save_path=args.save_path)

    # Optionally compute errors if true positions are available
    if "true_positions.npy" in args.positions_file:
        true_positions = np.load("data/true_positions.npy")
        predicted_positions = positions[-1]  # Last timestep positions
        mse = compute_error(true_positions, predicted_positions)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
