import pandas as pd
import torch
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from data import get_vocab, precompute_embeddings, MultiHotDataset, MultiHotCollator
from model import LogisticRegression
from train import train_model
from bootstrapping import bootstrap

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
    val_tokenised = [word_tokenize(str(review)) for review in X_val]
    test_tokenised = [word_tokenize(str(review)) for review in X_test]
    vocab = get_vocab(train_tokenised)
    vocab_size = len(vocab)

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

    print("Training Logistic Regression...")
    model1 = LogisticRegression(vocab_size)
    train_model(model1, train_loader, val_loader, model_name='logreg.pt', device=device)
    
    # --- Model 2: Transformer ---

    print("\nPreparing transformer data...")
    tokeniser = AutoTokenizer.from_pretrained('roberta-large')
    transformer = AutoModel.from_pretrained('roberta-large').to(device)
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
    model2 = LogisticRegression(input_dim=1024)
    train_model(model2, train_loader, val_loader, model_name='transformer_head.pt', device=device)

    # --- Bootstrap ---
    print("Preparing to bootstrap...")
    model1.load_state_dict(torch.load('logreg.pt', weights_only=True))
    model2.load_state_dict(torch.load('transformer_head.pt', weights_only=True))

    logreg_test = MultiHotDataset(test_tokenised, y_test, vocab)
    collator = MultiHotCollator(vocab_size)
    test_embeddings, test_labels = precompute_embeddings(X_test, y_test, tokeniser, transformer, device)
    transformer_test = TensorDataset(test_embeddings, test_labels)

    diffs, lower, upper = bootstrap(
        logreg_data=logreg_test,
        transformer_data=transformer_test,
        model1=model1, 
        model2=model2, 
        collator=collator,
        device=device)

    # --- Results ---
    mean_diff = np.mean(diffs)
    p_value = np.mean(np.array(diffs) <= 0)
    print(f"AUC difference (Model 2 - Model 1): {mean_diff:.3f} (95% CI: {lower:.3f} to {upper:.3f}), p={p_value:.3f}")