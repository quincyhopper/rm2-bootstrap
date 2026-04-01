import torch
from torch.utils.data import Dataset
from collections import Counter

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

@torch.no_grad()
def precompute_embeddings(X: list, y: list, tokeniser, model, device):
    model.eval()

    X = [str(x) for x in X]
    
    tokens = tokeniser(
        X,
        truncation=True, 
        padding='max_length',
        max_length=512,
        return_tensors='pt'
        )
    
    all_embeddings = []
    for i in range(0, len(X), 256):
        input_ids = tokens['input_ids'][i : i + 256].to(device)
        attention_mask = tokens['attention_mask'][i : i + 256].to(device)
        outputs = model(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings), torch.cat(y)