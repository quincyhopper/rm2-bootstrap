import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
from transformers import AutoModel

def get_vocab(tokenised_texts: list[list[str]]) -> dict:
    
    # Flatten
    tokens = [t.lower() for review in tokenised_texts for t in review]

    # Filter out infrequent tokens
    counts = Counter(tokens)
    vocab_list = [token for token, count in counts.items() if count >= 10]

    # Get token-idx map
    vocab_dict = {token: i for i, token in enumerate(vocab_list)}
    vocab_dict['UNK'] = len(vocab_list)

    return vocab_dict
    
class MultiHotDataset(Dataset):
    def __init__(self, tokenised_reviews: list[list[str]], targets: list[int], vocab: dict):
        super().__init__()

        self.tokenised_reviews = tokenised_reviews
        self.targets = targets
        self.vocab = vocab

    def __len__(self):
        return len(self.tokenised_reviews)

    def __getitem__(self, idx) -> torch.Tensor:
        indices = []
        for token in list(set(self.tokenised_reviews[idx])):
            result = self.vocab.get(token.lower())
            if result is not None:
                indices.append(result)
            else:
                indices.append(self.vocab['UNK'])

        return torch.tensor(indices), torch.tensor(self.targets[idx])
    
class MultiHotCollator():
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, batch: list[tuple[torch.Tensor], torch.Tensor]):

        x = torch.zeros(len(batch), self.vocab_size)
        for i, t in enumerate(batch):
            x[i].scatter_(0, t[0], 1.0)
        
        y = torch.stack([t[1] for t in batch]).long()

        return x, y
    
class EarlyStopping:
    def __init__(self, patience: int, model_name: str, delta: int=1e-4):
        self.patience = patience
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.count = 0
        self.delta = delta
        self.model_name = model_name

    def step(self, model, auc: float, epoch: int ):
        if auc > self.best_score + self.delta:
            self.best_score = auc
            self.count = 0
            self.best_epoch = epoch
            model.save(self.model_name)
        else:
            self.count += 1

        return self.count >= self.patience

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

        embedding = outputs.last_hidden_state[:, 0, :]

        return self.fc1(embedding)
    
    def save(self, path):
        torch.save(self.fc1.state_dict(), path)
    
class TransformerDataset(Dataset):
    def __init__(self, reviews, targets, tokeniser):
        super().__init__()

        self.encoded_texts = tokeniser(
            reviews,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return {k: v[index] for k, v in self.encoded_texts.items()}, self.targets[index]