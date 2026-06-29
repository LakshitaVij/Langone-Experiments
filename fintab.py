
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
    parser = argparse.ArgumentParser(description="Train/cross-validate the csPCa model")
    parser.add_argument("--input", default="cspca_features.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", default="cspca_model.pkl")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: could not find file at {args.input}", file=sys.stderr)
        sys.exit(1)

    feature_columns = [c for c in df.columns if c not in ID_AND_METADATA_COLUMNS]
    print(f"Using {len(feature_columns)} features: {feature_columns}")

    X = df[feature_columns]
    y = df["csPCa"].astype(int)
    groups = df["PatientID"]

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = HistGradientBoostingClassifier(
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=args.seed,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], proba)
        fold_aucs.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f} "
              f"(train n={len(train_idx)}, test n={len(test_idx)})")

    fold_aucs = np.array(fold_aucs)
    print(f"\nMean AUC: {fold_aucs.mean():.4f} (+/- {fold_aucs.std():.4f})")

    # HistGradientBoostingClassifier has no built-in feature_importances_, so use
    # permutation importance (fit on everything -- this is a sanity check, not a
    # held-out evaluation).
    final_model = HistGradientBoostingClassifier(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "feature_columns": feature_columns}, args.model_output)
    print(f"\nSaved trained model to {args.model_output}")

    result = permutation_importance(
        final_model, X, y, scoring="roc_auc", n_repeats=10, random_state=args.seed
    )
    ranked = sorted(zip(X.columns, result.importances_mean), key=lambda t: -t[1])
    print("\nPermutation importances (full-data fit, for a sanity check only):")
    for name, imp in ranked:
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()

