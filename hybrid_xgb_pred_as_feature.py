"""
Option 1: MLP + XGBoost Prediction as Feature

Append XGBoost's predicted probability as a 38th feature.
Train MLP encoder (38 → 128 → 64 → 2) on train set.
Evaluate on val set against baseline MLP (37 → 128 → 64 → 2).

Usage:
    python3 hybrid_xgb_pred_as_feature.py

Outputs:
    - xgb_pred_as_feature_metrics.txt (AUC comparison)
    - xgb_pred_as_feature_val_preds.csv (per-patient predictions)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──── PATHS ────────────────────────────────────────────────────────────────
TABULAR_MODEL   = "/gpfs/data/prostatelab/Lakshita/cspca_model_v4_trainonly.pkl"
TRAIN_CSV       = "/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/train_split_April4_testpatient_removed_relabeled.csv"
VAL_CSV         = "/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/val_split_April4_testpatient_removed_relabeled.csv"
TABULAR_FEATS   = "/gpfs/data/prostatelab/Lakshita/cspca_features_v4.csv"
OUTPUT_DIR      = "/gpfs/data/prostatelab/Lakshita/hybrid_models/"
METRICS_FILE    = f"{OUTPUT_DIR}/xgb_pred_as_feature_metrics.txt"
PREDS_FILE      = f"{OUTPUT_DIR}/xgb_pred_as_feature_val_preds.csv"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──── METADATA TO DROP ──────────────────────────────────────────────────────
ID_AND_METADATA = [
    "AccessionNumber", "PatientID", "split", "csPCa",
    "MaxGradeGroup", "MaxGleasonScore",
]

# ──── LOAD DATA ──────────────────────────────────────────────────────────────
print("[1/5] Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
tabular_df = pd.read_csv(TABULAR_FEATS)

# Merge to get labels
train_df = train_df[["AccessionNumber", "csPCa"]].merge(
    tabular_df, on="AccessionNumber", how="left"
)
val_df = val_df[["AccessionNumber", "csPCa"]].merge(
    tabular_df, on="AccessionNumber", how="left"
)

# Feature columns (drop metadata)
feature_cols = [c for c in train_df.columns if c not in ID_AND_METADATA]
print(f"  Train: {len(train_df)} rows, {len(feature_cols)} features")
print(f"  Val:   {len(val_df)} rows")

# ──── LOAD XGBOOST MODEL ────────────────────────────────────────────────────
print("[2/5] Loading XGBoost model...")
with open(TABULAR_MODEL, 'rb') as f:
    xgb_model = pickle.load(f)

# Get XGBoost predictions for train and val
X_train = train_df[feature_cols].fillna(0).values
X_val = val_df[feature_cols].fillna(0).values

xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]  # prob of csPCa
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

print(f"  XGBoost train probs: {xgb_train_probs.min():.3f} - {xgb_train_probs.max():.3f}")
print(f"  XGBoost val probs:   {xgb_val_probs.min():.3f} - {xgb_val_probs.max():.3f}")

# ──── ADD XGB PROB AS 38TH FEATURE ──────────────────────────────────────────
print("[3/5] Creating augmented feature matrices...")
X_train_aug = np.hstack([X_train, xgb_train_probs.reshape(-1, 1)])  # (N, 38)
X_val_aug = np.hstack([X_val, xgb_val_probs.reshape(-1, 1)])        # (N, 38)

y_train = train_df["csPCa"].values
y_val = val_df["csPCa"].values

print(f"  Train augmented: {X_train_aug.shape}")
print(f"  Val augmented:   {X_val_aug.shape}")

# ──── NORMALIZE ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_aug = scaler.fit_transform(X_train_aug)
X_val_aug = scaler.transform(X_val_aug)

# ──── DEFINE MLP ENCODER ────────────────────────────────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=38):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ──── DATASET ───────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ──── TRAIN ─────────────────────────────────────────────────────────────────
print("[4/5] Training MLP encoder (38 dims)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPEncoder(input_dim=38).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataset = TabularDataset(X_train_aug, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TabularDataset(X_val_aug, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_auc = 0.0
patience = 10
patience_count = 0

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_probs = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_probs.append(probs)
    
    val_probs = np.concatenate(val_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: val_auc={val_auc:.4f}")
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_count = 0
        best_probs = val_probs
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# ──── EVALUATION ────────────────────────────────────────────────────────────
print("[5/5] Evaluating...")

# Get best predictions
best_preds_prob = best_probs
best_preds_class = (best_preds_prob > 0.5).astype(int)

auc_hybrid = roc_auc_score(y_val, best_preds_prob)
print(f"\n{'='*60}")
print(f"OPTION 1: XGBoost Prediction as 38th Feature")
print(f"{'='*60}")
print(f"Val AUC (MLP + XGB pred):  {auc_hybrid:.4f}")
print(f"Best epoch AUC:            {best_val_auc:.4f}")

# Save predictions
output_df = pd.DataFrame({
    'AccessionNumber': val_df['AccessionNumber'].values,
    'csPCa_true': y_val,
    'csPCa_pred_prob': best_preds_prob,
    'csPCa_pred_class': best_preds_class,
})
output_df.to_csv(PREDS_FILE, index=False)
print(f"\nPredictions saved to: {PREDS_FILE}")

# Save metrics
with open(METRICS_FILE, 'w') as f:
    f.write(f"OPTION 1: XGBoost Prediction as 38th Feature\n")
    f.write(f"{'='*60}\n")
    f.write(f"Val AUC (MLP + XGB pred): {auc_hybrid:.4f}\n")
    f.write(f"Samples: {len(y_val)}\n")
    f.write(f"Positives (csPCa): {y_val.sum()}\n")
    f.write(f"Negatives: {(1 - y_val).sum()}\n")

print(f"Metrics saved to: {METRICS_FILE}")
print("\nDone!")
