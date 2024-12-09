# src/simulation.py
import numpy as np
from .mpi_communication import mpi_exchange_data, mpi_force_computation
from .force_prediction import load_model, predict_forces

def run_simulation(initial_positions, model_path, steps, force_function):
    positions = initial_positions
    model = load_model(model_path, input_dim=positions.shape[1], hidden_dim=64, output_dim=positions.shape[1])

    for step in range(steps):
        print(f"Step {step + 1}/{steps}")

        # Exchange data between MPI processes
        shared_positions = mpi_exchange_data(positions)

        # Compute forces using the ML model
        forces = mpi_force_computation(lambda x: predict_forces(model, x), shared_positions)

        # Update positions using forces (simple velocity-Verlet integration)
        positions += 0.01 * forces  # 假設固定時間步長

    return positions
