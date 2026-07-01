"""Train tabular csPCa classifier on TRAIN SPLIT ONLY.
Evaluates on val split to give clean, honest numbers without leakage.
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ID_AND_METADATA_COLUMNS = [
    "AccessionNumber", "PatientID", "split", "csPCa",
    "MaxGradeGroup", "MaxGleasonScore",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cspca_features_v4.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", default="cspca_model_v4_trainonly.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Split into train and val
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")

    feature_columns = [c for c in df.columns if c not in ID_AND_METADATA_COLUMNS]

    non_numeric = train_df[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"ERROR: non-numeric feature columns: {non_numeric}", file=sys.stderr)
        sys.exit(1)

    print(f"Using {len(feature_columns)} features")

    X_train = train_df[feature_columns]
    y_train = train_df["csPCa"].astype(int)
    groups  = train_df["PatientID"]

    X_val = val_df[feature_columns]
    y_val = val_df["csPCa"].astype(int)

    # Cross-validate on train set only
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train, groups), start=1):
        model = HistGradientBoostingClassifier(
            max_iter=1000, early_stopping=True,
            n_iter_no_change=10, validation_fraction=0.1,
            random_state=args.seed,
        )
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = model.predict_proba(X_train.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[test_idx], proba)
        fold_aucs.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f}")

    fold_aucs = np.array(fold_aucs)
    print(f"\nTrain CV AUC: {fold_aucs.mean():.4f} (+/- {fold_aucs.std():.4f})")

    # Train final model on train set only
    final_model = HistGradientBoostingClassifier(
        max_iter=1000, early_stopping=True,
        n_iter_no_change=10, validation_fraction=0.1,
        random_state=args.seed,
    )
    final_model.fit(X_train, y_train)

    # Evaluate on val set (clean, no leakage)
    val_proba = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Val AUC (clean): {val_auc:.4f}")

    joblib.dump({"model": final_model, "feature_columns": feature_columns}, args.model_output)
    print(f"\nSaved model to {args.model_output}")

if __name__ == "__main__":
    main()
