import pandas as pd
import torch
import time
import nltk
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from nltk.tokenize import word_tokenize
from model import get_vocab, MultiHotDataset, MultiHotCollator, LogisticRegression, EarlyStopping, MLP

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 1000
N_BOOTSTRAP = 100

def train(train_loader, model, optimiser, criterion, device):
    model.train()

    epoch_loss = 0.0

    for batch, targets in train_loader:
        batch, targets = batch.to(device), targets.to(device)
        optimiser.zero_grad()
        logits = model(batch)
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
        batch, targets = batch.to(device), targets.to(device)
        logits = model(batch)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1] # Prob of positive class
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    final_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_targets, all_probs)

    return final_loss, auc

def train_model(model, train_loader, val_loader, model_name, device):
    model.train()

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, model_name=model_name)

    for epoch in range(MAX_EPOCHS):
        start = time.time()

        train_loss = train(train_loader, model, optimiser, criterion, device)
        val_loss, val_auc = val(val_loader, model, criterion, device)

        total_time = time.time() - start

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f} | Val AUC: {val_auc:.2f} | Time: {total_time:.2f} seconds")

        if early_stopping.step(model, val_auc, epoch+1):
            print(f"Early stopping triggered. Model saved at epoch {early_stopping.best_epoch} with {early_stopping.best_score:.2f} AUC.\n")
            break

def get_bootstrap_loader(dataset, batch_size: int, vocab_size: int):
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=MultiHotCollator(vocab_size),
        pin_memory=True
    )

def bootstrap(X, y, model1, model2, criterion, vocab, batch_size, device) -> tuple[list, float, float]:

    test_dataset = MultiHotDataset(X, y, vocab)
    vocab_size = len(vocab)

    diffs = []
    for _ in range(N_BOOTSTRAP):
        loader = get_bootstrap_loader(test_dataset, batch_size=batch_size, vocab_size=vocab_size)
        _, auc1 = val(loader, model1, criterion, device)
        _, auc2 = val(loader, model2, criterion, device)
        diffs.append(auc2 - auc1) # Measures if model 2 is better than model 1

    sorted_diffs = sorted(diffs)
    lower, upper = np.percentile(sorted_diffs, 2.5), np.percentile(sorted_diffs, 97.5)

    return diffs, lower, upper

if __name__ == "__main__":

    # Download punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # Read datat
    print("Loading data...")
    df = pd.read_csv('data/reviews.csv')

    # Extract list of reviews and labels
    texts = df.text.tolist()
    labels = df.sentiment.map({'positive': 1, 'negative': 0}).tolist()

    # Train/val split
    print("Splitting...")
    X_train, X_other, y_train, y_other = train_test_split(
        texts, labels,
        train_size=0.8,
        random_state=42, 
        stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_other, y_other,
        random_state=42,
        train_size=0.5,
        stratify=y_other
    )

    # Tokenise
    print("Tokenising...")
    train_tokenised = [word_tokenize(review) for review in X_train]
    val_tokenised = [word_tokenize(review) for review in X_val]
    test_tokenised = [word_tokenize(review) for review in X_test]

    # Get vocab
    vocab = get_vocab(train_tokenised)
    vocab_size = len(vocab)

    # Init loaders
    train_loader = DataLoader(
        MultiHotDataset(train_tokenised, y_train, vocab),
        batch_size=BATCH_SIZE,
        collate_fn=MultiHotCollator(vocab_size),
        pin_memory=True
    )

    val_loader = DataLoader(
        MultiHotDataset(val_tokenised, y_val, vocab),
        batch_size=BATCH_SIZE,
        collate_fn=MultiHotCollator(vocab_size),
        pin_memory=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Model 1: Logistic Regression ---
    print("Training Logistic Regression...")
    logreg = LogisticRegression(vocab_size)
    train_model(logreg, train_loader, val_loader, model_name='logreg.pt', device=device)
    
    # --- Model 2: MLP ---
    print("Training MLP...")
    mlp = MLP(vocab_size)
    train_model(mlp, train_loader, val_loader, model_name='mlp.pt', device=device)

    # --- Bootstrap ---
    print("Bootstrapping...")
    logreg.load_state_dict(torch.load('logreg.pt', weights_only=True))
    mlp.load_state_dict(torch.load('mlp.pt', weights_only=True))
    diffs, lower, upper = bootstrap(test_tokenised, y_test, logreg, mlp, criterion=torch.nn.CrossEntropyLoss(), vocab=vocab, batch_size=BATCH_SIZE, device=device)

    # --- Results ---
    mean_diff = np.mean(diffs)
    p_value = np.mean(np.array(diffs) <= 0)
    print(f"AUC difference (Model 2 - Model 1): {mean_diff:.3f} (95% CI: {lower:.3f} to {upper:.3f}), p={p_value:.3f}")