"""
Run both hybrid options and compare:
  - Option 1: MLP + XGBoost Prediction as 38th Feature
  - Option 2: MLP + SHAP Values as Input

Outputs comparison table with baseline MLP (37 raw features).

Usage:
    python3 run_hybrid_comparison.py
"""

import subprocess
import sys

print("="*70)
print("RUNNING HYBRID MLP × XGBOOST COMPARISON")
print("="*70)

# Option 1
print("\n[1/2] Running Option 1: XGBoost Prediction as 38th Feature...")
print("-"*70)
result1 = subprocess.run(
    ["python3", "hybrid_xgb_pred_as_feature.py"],
    capture_output=False
)
if result1.returncode != 0:
    print("ERROR: Option 1 failed")
    sys.exit(1)

# Option 2
print("\n[2/2] Running Option 2: SHAP Values as Input...")
print("-"*70)
result2 = subprocess.run(
    ["python3", "hybrid_shap_values_input.py"],
    capture_output=False
)
if result2.returncode != 0:
    print("ERROR: Option 2 failed")
    sys.exit(1)

# Read and display results
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

import os
OUTPUT_DIR = "/gpfs/data/prostatelab/Lakshita/hybrid_models/"

if os.path.exists(f"{OUTPUT_DIR}/xgb_pred_as_feature_metrics.txt"):
    with open(f"{OUTPUT_DIR}/xgb_pred_as_feature_metrics.txt") as f:
        print(f.read())

if os.path.exists(f"{OUTPUT_DIR}/shap_values_metrics.txt"):
    with open(f"{OUTPUT_DIR}/shap_values_metrics.txt") as f:
        print(f.read())

print("\n" + "="*70)
print("FILES SAVED TO:")
print("="*70)
print(f"Option 1 predictions: {OUTPUT_DIR}/xgb_pred_as_feature_val_preds.csv")
print(f"Option 2 predictions: {OUTPUT_DIR}/shap_values_val_preds.csv")
print(f"SHAP values (train):  {OUTPUT_DIR}/shap_values_train.npy")
print(f"SHAP values (val):    {OUTPUT_DIR}/shap_values_val.npy")
print("="*70)
