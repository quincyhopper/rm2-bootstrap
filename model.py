import torch
import torch.nn as nn
from transformers import AutoModel

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained('roberta-large')

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask)

        embedding = outputs.last_hidden_state[:, -1, :]

        return self.fc1(embedding)
    
    def save(self, path):
        torch.save(self.fc1.state_dict(), path)