import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────
PREDS_CSV      = "/gpfs/data/prostatelab/Lakshita/late_fusion_preds_v4_val.csv"

def bootstrap_auc(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        auc = roc_auc_score(y_true[idx], y_pred[idx])
        aucs.append(auc)
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return lower, upper

def main():
    df = pd.read_csv(PREDS_CSV)
    
    y = df["Targets"].values
    s_img = df["imaging_score"].values
    s_tab = df["tabular_score"].values
    s_weighted = df["fusion_weighted"].values
    s_lr = df["fusion_lr"].values
    strategies = {
        "Imaging only":    s_img,
        "Tabular only":    s_tab,
        "Fusion weighted": s_weighted,
        "Fusion logistic": s_lr,
    }
    
    print("="*60)
    print(f"{'Strategy':<25} {'AUC':>6}  {'95% CI'}")
    print("="*60)
    for name, s_pred in strategies.items():
        auc = roc_auc_score(y, s_pred)
        lower, upper = bootstrap_auc(y, s_pred)
        print(f"{name:<25} {auc:.4f}  ({lower:.4f} - {upper:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()