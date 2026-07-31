"""
Credit Card Fraud Detection with a Deep Autoencoder (PyTorch)
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
File used: creditcard.csv

Usage:
    python train.py --data_dir data --epochs 30
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Data utilities
def load_data(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "creditcard.csv"))
    return df


def prepare_splits(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("Class",)]

    # Standardize Amount and Time; V1..V28 are already PCA-scaled but we
    # standardize everything together for consistency and stable training.
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df["Class"].values

    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    # Train the autoencoder ONLY on normal transactions from the training split
    X_train_normal = X_train_full[y_train_full == 0]

    return (X_train_normal, X_val, y_val, X_test, y_test, feature_cols, scaler)


# Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 14),
            nn.ReLU(),
            nn.Linear(14, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 14),
            nn.ReLU(),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# Train / eval
def reconstruction_errors(model, X, device, batch_size=512):
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = torch.mean((batch - recon) ** 2, dim=1)
            errors.extend(err.cpu().numpy())
    return np.array(errors)


def find_best_threshold(errors, y_true):
    precisions, recalls, thresholds = precision_recall_curve(y_true, errors)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    return thresholds[best_idx], precisions, recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data(args.data_dir)
    print(f"Total transactions: {len(df)} | Fraud cases: {df['Class'].sum()}")

    X_train_normal, X_val, y_val, X_test, y_test, feature_cols, scaler = prepare_splits(df)
    print(f"Training on {len(X_train_normal)} normal transactions only.")

    input_dim = X_train_normal.shape[1]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = Autoencoder(input_dim, bottleneck=args.bottleneck).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} | reconstruction MSE: {avg_loss:.6f}")

    torch.save(model.state_dict(), "model.pt")

    # --- Threshold selection on validation set ---
    val_errors = reconstruction_errors(model, X_val, device)
    best_threshold, precisions, recalls = find_best_threshold(val_errors, y_val)
    print(f"\nSelected reconstruction-error threshold (F1-optimal on val): {best_threshold:.6f}")

    # --- Final test evaluation ---
    test_errors = reconstruction_errors(model, X_test, device)
    test_preds = (test_errors > best_threshold).astype(int)

    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)

    print("\n=== Test set results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix ([[TN, FP], [FN, TP]]):\n", cm)

    # Precision-Recall curve plot (on test set, full curve) 
    p_curve, r_curve, _ = precision_recall_curve(y_test, test_errors)
    plt.figure(figsize=(7, 6))
    plt.plot(r_curve, p_curve, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Autoencoder Fraud Detector (Test Set)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=150)
    print("\nSaved model to model.pt and precision-recall curve to pr_curve.png")


if __name__ == "__main__":
    main()
