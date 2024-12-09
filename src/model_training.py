# src/model_training.py
import torch
import torch.nn as nn
import torch.optim as optim

class ForcePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ForcePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model(positions, forces, model_save_path, epochs=100, learning_rate=0.001):
    input_dim = positions.shape[1]
    output_dim = forces.shape[1]
    hidden_dim = 64  # 可根據需要調整

    model = ForcePredictor(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(positions, dtype=torch.float32)
        targets = torch.tensor(forces, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
