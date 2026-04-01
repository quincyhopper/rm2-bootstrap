import pandas as pd
import torch
import nltk
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from data import get_vocab, make_transformer_split, MultiHotDataset, MultiHotCollator, TransformerDataset
from model import LogisticRegression, Transformer
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
    logreg = LogisticRegression(vocab_size)
    train_model(logreg, train_loader, val_loader, model_name='logreg.pt', device=device)
    
    # --- Model 2: Transformer ---

    print("\nPreparing transformer data...")
    tokeniser = AutoTokenizer.from_pretrained('roberta-large')
    transformer_train = make_transformer_split(X_train, y_train, tokeniser)
    transformer_val = make_transformer_split(X_val, y_val, tokeniser)
    
    train_loader = DataLoader(
        TransformerDataset(transformer_train),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TransformerDataset(transformer_val),
        batch_size=256,
        pin_memory=True,
    )

    print("Training Transformer...")
    transformer = Transformer()
    train_model(transformer, train_loader, val_loader, model_name='transformer_head.pt', device=device)

    # --- Bootstrap ---
    print("Preparing to bootstrap...")
    logreg.load_state_dict(torch.load('logreg.pt', weights_only=True))
    transformer.fc1.load_state_dict(torch.load('transformer_head.pt', weights_only=True))

    logreg_test = MultiHotDataset(test_tokenised, y_test, vocab)
    collator = MultiHotCollator(vocab_size)
    transformer_test = TransformerDataset(
        make_transformer_split(X_test, y_test, tokeniser)
    )

    diffs, lower, upper = bootstrap(
        logreg_data=logreg_test,
        transformer_data=transformer_test,
        model1=logreg, 
        model2=transformer, 
        tokeniser=tokeniser, 
        vocab=vocab, 
        device=device)

    # --- Results ---
    mean_diff = np.mean(diffs)
    p_value = np.mean(np.array(diffs) <= 0)
    print(f"AUC difference (Model 2 - Model 1): {mean_diff:.3f} (95% CI: {lower:.3f} to {upper:.3f}), p={p_value:.3f}")