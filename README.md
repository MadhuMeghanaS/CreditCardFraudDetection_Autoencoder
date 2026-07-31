# Credit Card Fraud Detection with a Deep Autoencoder (PyTorch)

 Train an autoencoder to reconstruct *normal* transactions well; fraudulent transactions, which the model has never seen, get reconstructed poorly,giving a reconstruction-error score used to flag fraud.

## Dataset

**Kaggle: "Credit Card Fraud Detection"**
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- 284,807 European card transactions from September 2013, 492 of which are
  fraud (highly imbalanced — 0.17% fraud rate).
- Features `V1`...`V28` are PCA-anonymized for privacy; `Time` and `Amount`
  are raw; `Class` is the label (1 = fraud, 0 = normal) — used **only for
  evaluation**, never for training.

### Download instructions
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```
You should end up with `data/creditcard.csv`.


## Folder structure
```
project3_fraud_autoencoder/
├── data/                 
├── train.py              
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python train.py --data_dir data --epochs 30
```

The script will:
1. Load the data and standard-scale `Amount` and `Time` (the `V*` columns are
   already PCA-scaled).
2. Split into train/val/test, but **train only on normal (Class=0)
   transactions** from the training split — the model never sees fraud
   during training.
3. Train a symmetric autoencoder (28 → 16 → 8 → 16 → 28-ish bottleneck) to
   minimize reconstruction MSE on normal transactions.
4. Compute reconstruction error on the validation set (which contains both
   classes) and pick a threshold that maximizes F1.
5. Apply that threshold to the held-out test set and report precision,
   recall, F1, and a confusion matrix, plus a **precision-recall curve**
   (more informative than ROC-AUC for this level of imbalance) saved to
   `pr_curve.png`.
6. Save the trained model to `model.pt`.


## Future Enhancements
- Compare against Isolation Forest / One-Class SVM baselines on the same
  split — great "why deep learning" narrative.
- Add a variational autoencoder (VAE) and compare anomaly scores.
- Try a semi-supervised approach: use the tiny number of known fraud cases
  to tune the decision threshold via a small validation set (already partly
  done here) versus a purely unsupervised percentile-based threshold.
- Plot the 2D latent space (bottleneck) colored by true label to visually
  show separation.
