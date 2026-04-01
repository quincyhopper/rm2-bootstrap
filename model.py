import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)