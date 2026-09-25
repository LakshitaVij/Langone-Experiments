"""
Option 2: MLP + SHAP Values as Features

Replace raw features with XGBoost SHAP values.
SHAP values represent each feature's contribution to the model's prediction,
making them more informative than raw feature values.

Train MLP encoder (37 → 128 → 64 → 2) on SHAP values.
Evaluate on val set against baseline MLP and Option 1.

Usage:
    python3 hybrid_shap_values_input.py

Outputs:
    - shap_values_metrics.txt (AUC comparison)
    - shap_values_val_preds.csv (per-patient predictions)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# ──── PATHS ────────────────────────────────────────────────────────────────
TABULAR_MODEL   = "/gpfs/data/prostatelab/Lakshita/cspca_model_v4_trainonly.pkl"
TRAIN_CSV       = "/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/train_split_April4_testpatient_removed_relabeled.csv"
VAL_CSV         = "/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/val_split_April4_testpatient_removed_relabeled.csv"
TABULAR_FEATS   = "/gpfs/data/prostatelab/Lakshita/cspca_features_v4.csv"
OUTPUT_DIR      = "/gpfs/data/prostatelab/Lakshita/hybrid_models/"
METRICS_FILE    = f"{OUTPUT_DIR}/shap_values_metrics.txt"
PREDS_FILE      = f"{OUTPUT_DIR}/shap_values_val_preds.csv"
SHAP_TRAIN_FILE = f"{OUTPUT_DIR}/shap_values_train.npy"
SHAP_VAL_FILE   = f"{OUTPUT_DIR}/shap_values_val.npy"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──── METADATA TO DROP ──────────────────────────────────────────────────────
ID_AND_METADATA = [
    "AccessionNumber", "PatientID", "split", "csPCa",
    "MaxGradeGroup", "MaxGleasonScore",
]

# ──── LOAD DATA ──────────────────────────────────────────────────────────────
print("[1/6] Loading data...")
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
print("[2/6] Loading XGBoost model...")
with open(TABULAR_MODEL, 'rb') as f:
    xgb_model = pickle.load(f)

X_train = train_df[feature_cols].fillna(0).values
X_val = val_df[feature_cols].fillna(0).values

y_train = train_df["csPCa"].values
y_val = val_df["csPCa"].values

print(f"  Features: {len(feature_cols)}")

# ──── COMPUTE SHAP VALUES ──────────────────────────────────────────────────
print("[3/6] Computing SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(xgb_model)

# SHAP for training set
print("  Computing SHAP for train set...")
shap_train = explainer.shap_values(X_train)
if isinstance(shap_train, list):  # Binary classification returns [shap_neg, shap_pos]
    shap_train = shap_train[1]  # Use positive class SHAP values
shap_train = np.array(shap_train)  # (N_train, 37)

# SHAP for validation set
print("  Computing SHAP for val set...")
shap_val = explainer.shap_values(X_val)
if isinstance(shap_val, list):
    shap_val = shap_val[1]  # Use positive class SHAP values
shap_val = np.array(shap_val)  # (N_val, 37)

print(f"  SHAP train shape: {shap_train.shape}")
print(f"  SHAP val shape:   {shap_val.shape}")
print(f"  SHAP train range: {shap_train.min():.3f} - {shap_train.max():.3f}")
print(f"  SHAP val range:   {shap_val.min():.3f} - {shap_val.max():.3f}")

# Save for reference
np.save(SHAP_TRAIN_FILE, shap_train)
np.save(SHAP_VAL_FILE, shap_val)
print(f"  SHAP values saved to {OUTPUT_DIR}")

# ──── NORMALIZE ─────────────────────────────────────────────────────────────
print("[4/6] Normalizing SHAP values...")
scaler = StandardScaler()
shap_train_norm = scaler.fit_transform(shap_train)
shap_val_norm = scaler.transform(shap_val)

# ──── DEFINE MLP ENCODER ────────────────────────────────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=37):
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
print("[5/6] Training MLP encoder on SHAP values (37 dims)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPEncoder(input_dim=37).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataset = TabularDataset(shap_train_norm, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TabularDataset(shap_val_norm, y_val)
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
print("[6/6] Evaluating...")

# Get best predictions
best_preds_prob = best_probs
best_preds_class = (best_preds_prob > 0.5).astype(int)

auc_shap = roc_auc_score(y_val, best_preds_prob)
print(f"\n{'='*60}")
print(f"OPTION 2: SHAP Values as MLP Input")
print(f"{'='*60}")
print(f"Val AUC (MLP + SHAP):      {auc_shap:.4f}")
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
    f.write(f"OPTION 2: SHAP Values as MLP Input\n")
    f.write(f"{'='*60}\n")
    f.write(f"Val AUC (MLP + SHAP):     {auc_shap:.4f}\n")
    f.write(f"Samples: {len(y_val)}\n")
    f.write(f"Positives (csPCa): {y_val.sum()}\n")
    f.write(f"Negatives: {(1 - y_val).sum()}\n")

print(f"Metrics saved to: {METRICS_FILE}")
print("\nDone!")
