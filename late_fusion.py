"""
Probability-level late fusion: combine a separately-trained imaging model's
per-patient predictions with a separately-trained tabular model's
predictions, and compare simple-average / weighted-average / learned
logistic-regression blending against each alone.

This is a different fusion strategy from the three architectures train.py
trains (which fuse imaging + clinical *embeddings* inside one joint network).
Here, both models are trained independently and only their output
probabilities are combined afterward.

Inputs:
    --imaging-preds   Per-patient predictions CSV, in the format
                       src/metrics/plot.py's save_preds() writes (used by
                       train.py's trainer.test() when debugging.debug=true):
                       columns AccessionNumber, Predictions ('[p_neg, p_pos]'
                       string), Targets.
    --tabular-model   A joblib-dumped {"model": <sklearn estimator>,
                       "feature_columns": [...]} dict, trained separately on
                       the clinical features CSV.
    --tabular-features, --val-split-csv, --output-csv

Usage:
    python late_fusion.py \
        --imaging-preds /path/to/preds_epoch_N.csv \
        --tabular-features /path/to/cspca_features_v4.csv \
        --tabular-model /path/to/cspca_model_v4_trainonly.pkl \
        --val-split-csv /path/to/val_split.csv \
        --output-csv late_fusion_preds.csv
"""
import argparse
import ast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ID_AND_METADATA = [
    "AccessionNumber",
    "PatientID",
    "split",
    "csPCa",
    "MaxGradeGroup",
    "MaxGleasonScore",
]


def parse_imaging_prob(pred_str):
    """Parse a '[prob_neg, prob_pos]' string into the positive-class prob."""
    vals = ast.literal_eval(pred_str)
    return float(vals[1])


def main():
    parser = argparse.ArgumentParser(description="Probability-level late fusion")
    parser.add_argument(
        "--imaging-preds",
        default="/gpfs/data/prostatelab/Lakshita/experiments/cspca/eval/preds_epoch_tensor(0.8417).csv",
        help="Per-patient imaging model predictions CSV (Predictions/Targets/AccessionNumber)",
    )
    parser.add_argument(
        "--tabular-features",
        default="/gpfs/data/prostatelab/Lakshita/cspca_features_v4.csv",
        help="Clinical/tabular features CSV, keyed by AccessionNumber",
    )
    parser.add_argument(
        "--tabular-model",
        default="/gpfs/data/prostatelab/Lakshita/cspca_model_v4_trainonly.pkl",
        help="joblib-dumped {'model', 'feature_columns'} dict for the tabular-only model",
    )
    parser.add_argument(
        "--val-split-csv",
        default="/gpfs/data/prostatelab/processed_data/csv/data_split/pj_splits/val_split_April4_testpatient_removed_relabeled.csv",
        help="CSV listing the AccessionNumbers to evaluate on",
    )
    parser.add_argument(
        "--output-csv",
        default="late_fusion_preds.csv",
        help="Where to save per-patient scores for all fusion strategies",
    )
    args = parser.parse_args()

    print("Loading split set...")
    split_accessions = set(
        pd.read_csv(args.val_split_csv)["AccessionNumber"].astype(str).unique()
    )
    print(f"  Split patients: {len(split_accessions)}")

    print("Loading imaging predictions...")
    img_df = pd.read_csv(args.imaging_preds)
    img_df["imaging_score"] = img_df["Predictions"].apply(parse_imaging_prob)
    img_df = img_df[["AccessionNumber", "imaging_score", "Targets"]].copy()
    img_df["AccessionNumber"] = img_df["AccessionNumber"].astype(str)
    img_df = img_df[img_df["AccessionNumber"].isin(split_accessions)].copy()
    print(f"  Imaging predictions (split overlap): {len(img_df)}")

    print("Loading tabular features...")
    tab_df = pd.read_csv(args.tabular_features)
    tab_df["AccessionNumber"] = tab_df["AccessionNumber"].astype(str)
    tab_df = tab_df[tab_df["AccessionNumber"].isin(split_accessions)].copy()
    print(f"  Tabular features (split): {len(tab_df)}")

    feature_cols = [c for c in tab_df.columns if c not in ID_AND_METADATA]
    X_tab = tab_df[feature_cols].copy()

    print("Loading tabular model and running inference...")
    model_dict = joblib.load(args.tabular_model)
    tab_model = model_dict["model"]
    trained_cols = model_dict["feature_columns"]
    X_tab = X_tab.reindex(columns=trained_cols, fill_value=0)
    tab_df = tab_df.copy()
    tab_df["tabular_score"] = tab_model.predict_proba(X_tab)[:, 1]

    merged = img_df.merge(
        tab_df[["AccessionNumber", "tabular_score"]], on="AccessionNumber", how="inner"
    )
    print(f"  Merged patients: {len(merged)}")

    y = merged["Targets"].astype(int).values
    s_img = merged["imaging_score"].values
    s_tab = merged["tabular_score"].values

    # Strategy A: simple average
    s_avg = (s_img + s_tab) / 2

    # Strategy B: weighted average (grid search the imaging weight)
    best_w, best_auc_w = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        s_w = w * s_img + (1 - w) * s_tab
        a = roc_auc_score(y, s_w)
        if a > best_auc_w:
            best_auc_w, best_w = a, w
    s_weighted = best_w * s_img + (1 - best_w) * s_tab

    # Strategy C: learned logistic regression (5-fold CV to avoid overfitting)
    X_meta = np.column_stack([s_img, s_tab])
    X_scaled = StandardScaler().fit_transform(X_meta)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_lr = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X_scaled, y):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_scaled[train_idx], y[train_idx])
        s_lr[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]

    results = {
        "Imaging only": roc_auc_score(y, s_img),
        "Tabular only": roc_auc_score(y, s_tab),
        "Fusion: simple average": roc_auc_score(y, s_avg),
        f"Fusion: weighted (img={best_w:.2f})": roc_auc_score(y, s_weighted),
        "Fusion: logistic (CV)": roc_auc_score(y, s_lr),
    }

    print("\n" + "=" * 60)
    print(f"{'Strategy':<45} {'AUC':>10}")
    print("=" * 60)
    for name, auc in results.items():
        print(f"{name:<45} {auc:.4f}")
    print("=" * 60)

    merged["fusion_avg"] = s_avg
    merged["fusion_weighted"] = s_weighted
    merged["fusion_lr"] = s_lr
    merged.to_csv(args.output_csv, index=False)
    print(f"\nPer-patient predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
