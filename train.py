import torch
from sklearn.metrics import roc_auc_score

def train(train_loader, model, optimiser, criterion, device):
    model.train()

    epoch_loss = 0.0

    for batch, targets in train_loader:
        optimiser.zero_grad()
        targets = targets.to(device)
        logits = model(batch.to(device)).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        epoch_loss += loss.item()
        optimiser.step()

    return epoch_loss / len(train_loader)

@torch.no_grad()
def val(val_loader, model, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    for batch, targets in val_loader:
        targets = targets.to(device)
        logits = model(batch.to(device)).squeeze(1)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    final_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_targets, all_probs)

    return final_loss, auc

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

def train_model(model, train_loader, val_loader, model_name, device):

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()
    early_stopping = EarlyStopping(patience=10, model_name=model_name)

    for epoch in range(1000):
        train_loss = train(train_loader, model, optimiser, criterion, device)
        val_loss, val_auc = val(val_loader, model, criterion, device)

        print(f"Epoch [{epoch+1}/{1000}] | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f} | Val AUC: {val_auc:.2f}")

        if early_stopping.step(model, val_auc, epoch+1):
            print(f"Early stopping triggered. Model saved at epoch {early_stopping.best_epoch} with {early_stopping.best_score:.2f} AUC.\n")
            break