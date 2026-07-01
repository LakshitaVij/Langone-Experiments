"""
Late fusion: combine imaging model predictions with tabular model predictions.
Evaluates imaging-only, tabular-only, and three fusion strategies on VAL SET.

v4 update:
  - Uses cspca_model_v4.pkl (3 lesion slots, improved zone parsing, 37 features)
  - Uses cspca_features_v4.csv (exclusion criteria applied, 8172 rows)
  - Filters to VAL SET for iterative evaluation (reserve test set for final results)

Usage:
    python3 late_fusion_v4_val.py

Outputs:
    - AUC comparison table
    - late_fusion_preds_v4_val.csv (per-patient scores for all strategies)
"""

import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
IMAGING_PREDS   = "/gpfs/data/prostatelab/Lakshita/experiments/cspca/eval/preds_epoch_tensor(0.8417).csv"
TABULAR_FEATS   = "/gpfs/data/prostatelab/Lakshita/cspca_features_v4.csv"
TABULAR_MODEL   = "/gpfs/data/prostatelab/Lakshita/cspca_model_v4.pkl"
VAL_SET_CSV     = "/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/val_split_April4_testpatient_removed_relabeled.csv"
OUTPUT_CSV      = "/gpfs/data/prostatelab/Lakshita/late_fusion_preds_v4_val.csv"

# ── columns to drop before tabular inference (same as training) ─────────────
ID_AND_METADATA = [
    "AccessionNumber", "PatientID", "split", "csPCa",
    "MaxGradeGroup", "MaxGleasonScore",
]

def parse_imaging_prob(pred_str):
    """Parse '[prob_neg, prob_pos]' string → prob_pos float."""
    vals = ast.literal_eval(pred_str)
    return float(vals[1])

def main():
    # ── 0. load val set AccessionNumbers ────────────────────────────────────
    print("Loading val set...")
    val_df_raw = pd.read_csv(VAL_SET_CSV)
    val_accessions = set(val_df_raw["AccessionNumber"].astype(str).unique())
    print(f"  Val set patients: {len(val_accessions)}")

    # ── 1. load imaging predictions and filter to val ───────────────────────
    print("Loading imaging predictions...")
    img_df = pd.read_csv(IMAGING_PREDS) 
    img_df["imaging_score"] = img_df["Predictions"].apply(parse_imaging_prob) 
    img_df = img_df[["AccessionNumber", "imaging_score", "Targets"]].copy()
    img_df["AccessionNumber"] = img_df["AccessionNumber"].astype(str)
    
    # Filter to val set
    img_df = img_df[img_df["AccessionNumber"].isin(val_accessions)].copy()
    print(f"  Imaging predictions (val overlap): {len(img_df)}")

    # ── 2. load tabular features + run inference ────────────────────────────
    print("Loading tabular features (v4)...")
    tab_df = pd.read_csv(TABULAR_FEATS)
    tab_df["AccessionNumber"] = tab_df["AccessionNumber"].astype(str)

    # Filter to val set
    tab_df = tab_df[tab_df["AccessionNumber"].isin(val_accessions)].copy()
    print(f"  Tabular features (val set): {len(tab_df)}")

    # build feature matrix 
    feature_cols = [c for c in tab_df.columns if c not in ID_AND_METADATA]
    X_tab = tab_df[feature_cols].copy()

    print("Loading tabular model v4 and running inference...")
    with open(TABULAR_MODEL, "rb") as f:
        model_dict = pickle.load(f)
        tab_model = model_dict["model"]
        trained_cols = model_dict["feature_columns"]

    # align columns (fill missing with 0)
    X_tab = X_tab.reindex(columns=trained_cols, fill_value=0)

    tab_scores = tab_model.predict_proba(X_tab)[:, 1]
    tab_df = tab_df.copy()
    tab_df["tabular_score"] = tab_scores

    # ── 3. join on AccessionNumber ──────────────────────────────────────────
    merged = img_df.merge(
        tab_df[["AccessionNumber", "tabular_score"]],
        on="AccessionNumber",
        how="inner",
    )
    print(f"  Merged patients (val): {len(merged)}")

    y      = merged["Targets"].astype(int).values
    s_img  = merged["imaging_score"].values
    s_tab  = merged["tabular_score"].values

    # ── 4. fusion strategies ────────────────────────────────────────────────

    # Strategy A: simple average
    s_avg = (s_img + s_tab) / 2

    # Strategy B: weighted average (tune weight on same data — quick grid search)
    best_w, best_auc_w = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        s_w = w * s_img + (1 - w) * s_tab
        a   = roc_auc_score(y, s_w)
        if a > best_auc_w:
            best_auc_w, best_w = a, w
    s_weighted = best_w * s_img + (1 - best_w) * s_tab

    # Strategy C: learned logistic regression (5-fold CV to avoid overfitting)
    X_meta   = np.column_stack([s_img, s_tab])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_lr     = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X_scaled, y):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_scaled[train_idx], y[train_idx])
        s_lr[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]

    # ── 5. compute and print AUCs ───────────────────────────────────────────
    results = {
        "Imaging only":             roc_auc_score(y, s_img),
        "Tabular only (v4)":        roc_auc_score(y, s_tab),
        "Fusion: simple average":   roc_auc_score(y, s_avg),
        f"Fusion: weighted (img={best_w:.2f})": roc_auc_score(y, s_weighted),
        "Fusion: logistic (CV)":    roc_auc_score(y, s_lr),
    }

    print("\n" + "="*60)
    print(f"{'Strategy':<45} {'AUC':>10}")
    print("="*60)
    for name, auc in results.items():
        print(f"{name:<45} {auc:.4f}")
    print("="*60)
    print("\nComparison to prior late fusion (test set, v1 tabular):")
    print("  - Imaging only (test): 0.8329")
    print("  - Tabular v1 (test): 0.8720")
    print("  - Weighted fusion v1 (test): 0.8813")
    print("="*60)

    # ── 6. save per-patient predictions ────────────────────────────────────
    merged["fusion_avg"]      = s_avg
    merged["fusion_weighted"] = s_weighted
    merged["fusion_lr"]       = s_lr
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-patient predictions saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()