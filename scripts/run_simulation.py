from src.simulation import run_simulation
from src.force_prediction import load_model, predict_forces
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--steps", type=int, required=True, help="Number of simulation steps")
    parser.add_argument("--output_file", default="results/final_positions.npy", help="Path to save final positions")
    args = parser.parse_args()

    # Example initial particle positions
    num_particles = 100
    dimensions = 3
    initial_positions = np.random.rand(num_particles, dimensions)

    # Load the trained model
    model = load_model(args.model_path, input_dim=dimensions, hidden_dim=64, output_dim=dimensions)

    # Define the force function using the trained model
    def force_function(positions):
        return predict_forces(model, positions)

    # Run the simulation
    final_positions = run_simulation(initial_positions, args.model_path, args.steps, force_function)
    
    # Save all time steps
    all_positions = run_simulation(initial_positions, args.model_path, args.steps, force_function)
    np.save("results/all_positions.npy", all_positions)
    print("All positions saved to 'results/all_positions.npy'")

