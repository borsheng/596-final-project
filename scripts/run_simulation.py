# scripts/run_simulation.py
from src.simulation import run_simulation
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--steps", type=int, required=True, help="Number of simulation steps")
    args = parser.parse_args()

    # Example initial particle positions
    initial_positions = np.random.rand(100, 3)  # 100 particles in 3D space

    final_positions = run_simulation(initial_positions, args.model_path, args.steps)
    print("Final positions:", final_positions)
