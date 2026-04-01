import pandas as pd
import torch
import nltk
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from nltk.tokenize import word_tokenize
from model import get_vocab, MultiHotDataset, MultiHotCollator, LogisticRegression, EarlyStopping, Transformer, TransformerDataset

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 1000
N_BOOTSTRAP = 100

def forward_pass(model, input, device):
    if isinstance(input, dict):
        input_ids = input['input_ids'].to(device)
        attention_mask = input['attention_mask'].to(device)
        return model(input_ids, attention_mask)
    else:
        input = input.to(device)
        return model(input)

def train(train_loader, model, optimiser, criterion, device):
    model.train()

    epoch_loss = 0.0

    for batch, targets in train_loader:
        optimiser.zero_grad()
        targets = targets.to(device)
        logits = forward_pass(model, batch, device)
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
        logits = forward_pass(model, batch, device)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1] # Prob of positive class
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    final_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_targets, all_probs)

    return final_loss, auc

def train_model(model, train_loader, val_loader, model_name, device):

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, model_name=model_name)

    for epoch in range(MAX_EPOCHS):
        train_loss = train(train_loader, model, optimiser, criterion, device)
        val_loss, val_auc = val(val_loader, model, criterion, device)

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f} | Val AUC: {val_auc:.2f}")

        if early_stopping.step(model, val_auc, epoch+1):
            print(f"Early stopping triggered. Model saved at epoch {early_stopping.best_epoch} with {early_stopping.best_score:.2f} AUC.\n")
            break

def get_bootstrap_loader(dataset, batch_size: int, collator_fn=None):
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator_fn,
        pin_memory=True
    )

def bootstrap(X, y, logreg, transformer, tokenizer, criterion, vocab, batch_size, device) -> tuple[list, float, float]:

    # Prepare logreg
    test_tokenized = [word_tokenize(review) for review in X]
    logreg_data = MultiHotDataset(test_tokenized, y, vocab)
    collator = MultiHotCollator(vocab_size=len(vocab))

    # Prepare transformer
    transformer_data = TransformerDataset(X, y, tokenizer)

    diffs = []
    for _ in range(N_BOOTSTRAP):
        loader1 = get_bootstrap_loader(logreg_data, batch_size=batch_size, collator_fn=collator)
        loader2 = get_bootstrap_loader(transformer_data, batch_size=batch_size)
        _, auc1 = val(loader1, logreg, criterion, device)
        _, auc2 = val(loader2, transformer, criterion, device)
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

    # Read data
    print("Loading data...")
    df = pd.read_csv('data/Compiled_Reviews.txt', sep='\t')

    # Extract list of reviews and labels
    texts = df.text.tolist()
    labels = df.sentiment.map({'positive': 1, 'negative': 0}).tolist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # --- Model 1: Logistic Regression ---

    # Tokenise
    print("Tokenising...")
    train_tokenised = [word_tokenize(review) for review in X_train]
    val_tokenised = [word_tokenize(review) for review in X_val]

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

    print("Training Logistic Regression...")
    logreg = LogisticRegression(vocab_size)
    train_model(logreg, train_loader, val_loader, model_name='logreg.pt', device=device)
    
    # --- Model 2: Transformer ---

    tokeniser = AutoTokenizer.from_pretrained('roberta-large')

    train_loader = DataLoader(
        TransformerDataset(X_train, y_train, tokeniser),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        TransformerDataset(X_val, y_val, tokeniser),
        batch_size=BATCH_SIZE,
        pin_memory=True
    )

    print("Training Transformer...")
    transformer = Transformer()
    train_model(transformer, train_loader, val_loader, model_name='transformer.pt', device=device)

    # --- Bootstrap ---
    print("Bootstrapping...")
    logreg.load_state_dict(torch.load('logreg.pt', weights_only=True))
    transformer.fc1.load_state_dict(torch.load('transformer.pt', weights_only=True))

    diffs, lower, upper = bootstrap(
        X_test, y_test, logreg, 
        transformer, tokeniser, 
        criterion=torch.nn.CrossEntropyLoss(), 
        vocab=vocab, batch_size=BATCH_SIZE, 
        device=device)

    # --- Results ---
    mean_diff = np.mean(diffs)
    p_value = np.mean(np.array(diffs) <= 0)
    print(f"AUC difference (Model 2 - Model 1): {mean_diff:.3f} (95% CI: {lower:.3f} to {upper:.3f}), p={p_value:.3f}")