# src/simulation.py
import numpy as np
from .mpi_communication import mpi_exchange_data, mpi_force_computation
from .force_prediction import load_model, predict_forces

def run_simulation(initial_positions, model_path, steps, force_function):
    positions = initial_positions
    all_positions = [positions.copy()]  # Save initial positions

    for step in range(steps):
        print(f"Step {step + 1}/{steps}")
        forces = force_function(positions)
        positions += 0.01 * forces  # Update positions
        all_positions.append(positions.copy())  # Save positions at each step

    return np.array(all_positions)  # Return all positions

