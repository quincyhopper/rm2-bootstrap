import pandas as pd
import torch
import nltk
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from nltk.tokenize import word_tokenize

from data import get_vocab, precompute_embeddings, MultiHotDataset, MultiHotCollator
from model import LogisticRegression
from train import train_model, predict

def bootstrap(logreg_loader, transformer_loader, logreg_model, transformer, device):
    probs1, labels1 = predict(logreg_loader, logreg_model)
    probs2, labels2 = predict(transformer_loader, transformer, device)

    n = len(labels1)
    diffs = []

    for _ in range(1000):
        idx = np.random.randint(low=0, high=n, size=(n,)) # sample with replacement
        auc1 = roc_auc_score(labels1[idx], probs1[idx])
        auc2 = roc_auc_score(labels2[idx], probs2[idx])
        diffs.append(auc2 - auc1)

    diffs = np.array(diffs)
    observed_mean = np.mean(diffs)
    centered = diffs - observed_mean
    p_value = np.mean(centered <= -observed_mean)
    lower, upper = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

    return diffs, lower, upper, p_value

if __name__ == "__main__":

    # Download punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # Read data
    print("Loading data...")
    df = pd.read_csv('data/Compiled_Reviews.txt', sep='\t')
    df = df.dropna(subset=['REVIEW', 'RATING'])

    # Extract list of reviews and labels
    texts = df.REVIEW.tolist()
    labels = df.RATING.map({'positive': 1, 'negative': 0}).tolist()
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
        train_size=0.5,
        random_state=42,
        stratify=y_other
    )

    # --- Model 1: Logistic Regression ---

    # Tokenise
    print("Preparing logreg data...")
    train_tokenised = [word_tokenize(str(review)) for review in X_train]
    vocab = get_vocab(train_tokenised)
    vocab_size = len(vocab)
    model1 = LogisticRegression(vocab_size)

    if not os.path.exists('logreg.pt'):
        print("Training Logistic Regression...")
        val_tokenised = [word_tokenize(str(review)) for review in X_val]

        # Init loaders
        train_loader = DataLoader(
            MultiHotDataset(train_tokenised, y_train, vocab),
            batch_size=256,
            collate_fn=MultiHotCollator(vocab_size),
            pin_memory=True,
        )

        val_loader = DataLoader(
            MultiHotDataset(val_tokenised, y_val, vocab),
            batch_size=256,
            collate_fn=MultiHotCollator(vocab_size),
            pin_memory=True,
        )

        train_model(model1, train_loader, val_loader, model_name='logreg.pt', device=device)
    else:
        print("Found a trained logistic regression model. Skipping training")
    
    # --- Model 2: Transformer ---

    print("\nPreparing transformer data...")
    model2 = LogisticRegression(input_dim=1024)
    transformer = AutoModel.from_pretrained('roberta-large').to(device)
    if not os.path.exists('transformer_head.pt'):
        tokeniser = AutoTokenizer.from_pretrained('roberta-large')
        train_embeddings, train_labels = precompute_embeddings(X_train, y_train, tokeniser, transformer, device)
        val_embeddings, val_labels = precompute_embeddings(X_val, y_val, tokeniser, transformer, device)
        
        train_loader = DataLoader(
            TensorDataset(train_embeddings, train_labels),
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            TensorDataset(val_embeddings, val_labels),
            batch_size=256,
            pin_memory=True,
        )

        print("Training Transformer...")
        train_model(model2, train_loader, val_loader, model_name='transformer_head.pt', device=device)
    else:
        print("Found a trained transformer head. Skipping training.")

    # --- Bootstrap ---
    print("Preparing to bootstrap...")

    # Load models
    model1.load_state_dict(torch.load('logreg.pt', weights_only=True))
    model2.load_state_dict(torch.load('transformer_head.pt', weights_only=True))

    # Logreg 
    test_tokenised = [word_tokenize(str(review)) for review in X_test]
    model1_dataset = MultiHotDataset(test_tokenised, y_test, vocab)

    # Transformer
    test_embeddings, test_labels = precompute_embeddings(X_test, y_test, tokeniser, transformer, device)
    model2_dataset = TensorDataset(test_embeddings, test_labels)

    # Get loaders
    model1_loader = DataLoader(
        dataset=MultiHotDataset(test_tokenised, y_test, vocab), 
        batch_size=256, 
        collate_fn=MultiHotCollator(vocab_size),
        pin_memory=True
        )
    
    model2_loader = DataLoader(
        TensorDataset(val_embeddings, val_labels),
        batch_size=256,
        pin_memory=True,
        )
    
    diffs, lower, upper, p_value = bootstrap(model1_loader, model2_loader, model1, model2, device)
    print(f"CI: {lower:.3f} to {upper:.3f} | p-value {p_value}")

    pd.DataFrame(diffs, columns=['diffs']).to_csv('diffs.csv', index=False)