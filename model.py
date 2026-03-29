import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter

BATCH_SIZE = 256

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
        
        y = torch.stack([t[1] for t in batch])

        return x, y
    
class LogisticRegression(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.fc1 = nn.Linear(vocab_size, 2)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        return x
        


    





    

