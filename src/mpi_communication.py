# src/mpi_communication.py
from mpi4py import MPI
import numpy as np

def mpi_exchange_data(particle_positions):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_data = np.array_split(particle_positions, size)[rank]
    gathered_data = comm.allgather(local_data)
    combined_data = np.concatenate(gathered_data, axis=0)

    return combined_data

def mpi_force_computation(force_function, particle_positions):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_data = np.array_split(particle_positions, size)[rank]
    local_forces = force_function(local_data)

    all_forces = comm.allgather(local_forces)
    combined_forces = np.concatenate(all_forces, axis=0)

    return combined_forces
