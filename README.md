# Machine Learning-Based Force Fields for Molecular Dynamics

This repository demonstrates the use of machine learning (ML) force fields (MLFFs) to accelerate molecular dynamics (MD) simulations. By integrating ML models and MPI communications, we achieve faster and more efficient simulations.

---

## Features

- **Machine Learning**: Train a neural network model to predict forces for particles.
- **MPI Integration**: Use asynchronous MPI communications for parallel data exchange and computation.
- **Simulation**: Perform MD simulations with velocity-Verlet integration using ML-predicted forces.

---

## Project Structure

```
MLFF-MD-Simulation/
│
├── data/               # Contains datasets and parameters
│   ├── dataset.npz          # Training data (positions and forces)
│   ├── true_positions.npy   # Ground truth particle trajectories
│   ├── initial_positions.npy # Initial positions for simulation
│   ├── params.json          # Model hyperparameters
│
├── models/             # Stores trained machine learning models
│
├── src/                # Core implementation modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── force_prediction.py
│   ├── mpi_communication.py
│   └── simulation.py
│
├── scripts/            # Scripts for training, simulation, and analysis
│   ├── train_model.py
│   ├── run_simulation.py
│   └── analyze_results.py
│
├── results/            # Stores simulation results and analysis outputs
├── tests/              # Unit tests for the project
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- MPI (e.g., OpenMPI)
- Required Python libraries (listed in `requirements.txt`)

Install dependencies with the following command:

```bash
pip install -r requirements.txt
```

---

### Workflow

#### 1. Prepare Data

Store the dataset in the `data/` folder. Example structure for the training dataset (`dataset.npz`):

```python
{
    'positions': ndarray(shape=(N, D)),  # Particle positions
    'forces': ndarray(shape=(N, D))     # Corresponding forces
}
```

#### 2. Train the Model

Train a neural network to predict forces using the training dataset:

```bash
python scripts/train_model.py --data_path data/dataset.npz --save_path models/model.pth
```

#### 3. Run the Simulation

Simulate particle dynamics using the trained model:

```bash
mpiexec -n 4 python scripts/run_simulation.py --model_path models/model.pth --steps 100
```

#### 4. Analyze Results

Visualize particle trajectories or compute prediction errors:

```bash
python scripts/analyze_results.py --positions_file results/positions.npy --save_path results/trajectories.png
```

---

## Files in the `data/` Directory

- **`dataset.npz`**: Contains particle positions and forces for training.
- **`true_positions.npy`**: Ground truth particle trajectories for validation.
- **`initial_positions.npy`**: Initial positions for simulation.
- **`params.json`**: Hyperparameters for the machine learning model.

---

## Example Outputs

- **Simulation Results**:
  - Predicted particle positions over time, saved in the `results/` directory.
- **Trajectory Plot**:
  - Example output of particle trajectories:
    ![Particle Trajectories](results/trajectories.png)

---

## Testing

Run unit tests for the project:

```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## Contact

For questions or collaboration, feel free to contact the repository owner.

--- 
