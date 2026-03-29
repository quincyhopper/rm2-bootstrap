import pandas as pd
import torch
import time
import nltk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import get_vocab, MultiHotDataset, MultiHotCollator, LogisticRegression
from nltk.tokenize import word_tokenize

BATCH_SIZE = 64
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 1000

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

    for batch, targets in val_loader:
        batch, targets = batch.to(device), targets.to(device)
        logits = model(batch)
        loss = criterion(logits, targets)
        total_loss += loss.item()

    return total_loss / len(val_loader)

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

    # Get vocab
    vocab = get_vocab(train_tokenised)
    vocab_size = len(vocab)

    # Init model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LogisticRegression(vocab_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    # Init loaders
    train_loader = DataLoader(
        MultiHotDataset(train_tokenised, y_train, vocab),
        batch_size=BATCH_SIZE,
        collate_fn=MultiHotCollator(vocab_size),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        MultiHotDataset(val_tokenised, y_val, vocab),
        batch_size=BATCH_SIZE,
        collate_fn=MultiHotCollator(vocab_size),
        num_workers=4,
        pin_memory=True
    )

    print("Begin training...")
    for epoch in range(MAX_EPOCHS):
        start = time.time()

        train_loss = train(train_loader, model, optimiser, criterion, device)
        val_loss = val(val_loader, model, criterion, device)

        total_time = time.time() - start

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f} | Time: {total_time:.2f} seconds")