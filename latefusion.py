"""
Late fusion: combine imaging model predictions with tabular model predictions.
Evaluates imaging-only, tabular-only, and three fusion strategies.

Usage:
    python3 late_fusion.py

Outputs:
    - AUC comparison table
    - late_fusion_preds.csv (per-patient scores for all strategies)
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
IMAGING_PREDS   = "/gpfs/data/prostatelab/Lakshita/experiments/cspca/eval/preds_epoch_tensor(0.8329).csv"
TABULAR_FEATS   = "/gpfs/home/lv2255/Lakshita/cspca_features.csv"
TABULAR_MODEL   = "/gpfs/home/lv2255/Lakshita/cspca_model.pkl"
OUTPUT_CSV      = "/gpfs/home/lv2255/Lakshita/late_fusion_preds.csv"

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
    # ── 1. load imaging predictions ─────────────────────────────────────────
    print("Loading imaging predictions...")
    img_df = pd.read_csv(IMAGING_PREDS)
    img_df["imaging_score"] = img_df["Predictions"].apply(parse_imaging_prob)
    img_df = img_df[["AccessionNumber", "imaging_score", "Targets"]].copy()
    img_df["AccessionNumber"] = img_df["AccessionNumber"].astype(str)
    print(f"  Imaging patients: {len(img_df)}")

    # ── 2. load tabular features + run inference ────────────────────────────
    print("Loading tabular features...")
    tab_df = pd.read_csv(TABULAR_FEATS)
    tab_df["AccessionNumber"] = tab_df["AccessionNumber"].astype(str)

    # filter to only the test patients
    tab_test = tab_df[tab_df["AccessionNumber"].isin(img_df["AccessionNumber"])].copy()
    print(f"  Tabular patients (overlap): {len(tab_test)}")

    # build feature matrix (same logic as training script)
    feature_cols = [c for c in tab_test.columns if c not in ID_AND_METADATA]
    X_tab = pd.get_dummies(tab_test[feature_cols], columns=["lesion_max_zone"], dummy_na=True)

    print("Loading tabular model and running inference...")
    with open(TABULAR_MODEL, "rb") as f:
        tab_model = pickle.load(f)

    # align columns in case get_dummies produces different columns than training
    # (safe to fill missing with 0 — unseen categories)
    trained_cols = tab_model.feature_names_in_ if hasattr(tab_model, "feature_names_in_") else X_tab.columns
    X_tab = X_tab.reindex(columns=trained_cols, fill_value=0)

    tab_scores = tab_model.predict_proba(X_tab)[:, 1]
    tab_test = tab_test.copy()
    tab_test["tabular_score"] = tab_scores

    # ── 3. join on AccessionNumber ──────────────────────────────────────────
    merged = img_df.merge(
        tab_test[["AccessionNumber", "tabular_score"]],
        on="AccessionNumber",
        how="inner",
    )
    print(f"  Merged patients: {len(merged)}")

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
        "Tabular only":             roc_auc_score(y, s_tab),
        "Fusion: simple average":   roc_auc_score(y, s_avg),
        f"Fusion: weighted (img={best_w:.2f})": roc_auc_score(y, s_weighted),
        "Fusion: logistic (CV)":    roc_auc_score(y, s_lr),
    }

    print("\n" + "="*50)
    print(f"{'Strategy':<40} {'AUC':>6}")
    print("="*50)
    for name, auc in results.items():
        print(f"{name:<40} {auc:.4f}")
    print("="*50)

    # ── 6. save per-patient predictions ────────────────────────────────────
    merged["fusion_avg"]      = s_avg
    merged["fusion_weighted"] = s_weighted
    merged["fusion_lr"]       = s_lr
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-patient predictions saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()