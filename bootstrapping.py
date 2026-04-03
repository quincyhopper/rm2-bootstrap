import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from train import val

def get_bootstrap_loader(dataset, collator_fn=None):
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    return DataLoader(
        dataset,
        batch_size=256,
        sampler=sampler,
        collate_fn=collator_fn,
        pin_memory=True
    )

def bootstrap(logreg_data, transformer_data, model1, model2, collator, device) -> tuple[list, float, float]:
    criterion = nn.BCEWithLogitsLoss()

    diffs = []
    for i in range(200):
        print(f"Boostrapping ({i})")
        loader1 = get_bootstrap_loader(logreg_data, collator_fn=collator)
        loader2 = get_bootstrap_loader(transformer_data)
        _, auc1 = val(loader1, model1, criterion, device)
        _, auc2 = val(loader2, model2, criterion, device)
        diffs.append(auc2 - auc1) # Measures if model 2 is better than model 1

    sorted_diffs = sorted(diffs)
    lower, upper = np.percentile(sorted_diffs, 2.5), np.percentile(sorted_diffs, 97.5)

    return diffs, lower, upper