# src/force_prediction.py
import torch
import numpy as np
from .model_training import ForcePredictor

def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = ForcePredictor(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_forces(model, new_positions):
    inputs = torch.tensor(new_positions, dtype=torch.float32)
    with torch.no_grad():
        predicted_forces = model(inputs).numpy()
    return predicted_forces
