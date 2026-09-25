# Multimodal csPCa Detection — Fusion Architectures

Research code for detecting clinically significant prostate cancer (csPCa) by
fusing 3D MRI (T2 / ADC / B1500) with structured clinical features (PSA,
PI-RADS, lesion location, etc.) using a frozen ResNet3D imaging backbone.

This repo contains many historical experiment snapshots (files prefixed
`july15`/`july20`/`july29`/`aug11`). The three **final, documented**
architectures — the ones this README covers — are:

| Architecture | Model file | Config | Writeup |
|---|---|---|---|
| CBAM (channel + spatial clinical attention) | `aug11cbam.py` | `configs/cbam.yaml` | `cbam.md` |
| Early-scalar clinical attention | `aug11earlyscalar.py` | `configs/earlyscalar.yaml` | `earlyscalar.md` |
| Late fusion (flat clinical encoder) | `aug11latefusionflat.py` | `configs/latefusion_flat.yaml` | `latefusion.md` |

Read the `.md` writeup (and matching `.png` diagram) for the design rationale
behind each architecture. This README only covers how to actually run them.

## Setup

```bash
pip install -r requirements.txt
```

Everything here was run on the lab's HPC cluster (`prostatelab`), where the
H5 imaging data and CSV splits live under `/gpfs/data/prostatelab/...`. This
code will not do anything useful off that cluster (or an equivalent copy of
the data) — the configs point at real cluster paths that you'll need access
to.

## The `src/` package

`src/` (data loading, the ResNet3D backbone, metrics/plotting) mirrors code
from the lab's shared module of the same name. It's included directly in
this repo so training is actually runnable from here, rather than assuming
everyone has that shared module on their `PYTHONPATH` already. If the lab's
canonical `src/` has moved on since, treat this copy as a snapshot and diff
against the authoritative version before trusting it for new results.

Expected H5 file structure per exam (see `src/data/loader.py` docstring for
the full spec):
```
['axt2'], ['adc'], ['b1500']        # imaging volumes
.attrs['maxPIRADS'], .attrs['psa'], .attrs['prostate_volume'], ...
```
Clinical/tabular features come from a separate CSV (`data.tabular_csv` in
each config) keyed by `AccessionNumber`, with 37 feature columns matching
`FrozenClinicalEncoder`'s expected input size.

## Running training

Each architecture has its own config; they share the same entrypoint:

```bash
python train.py --config configs/cbam.yaml
python train.py --config configs/earlyscalar.yaml
python train.py --config configs/latefusion_flat.yaml
```

Before running, open the config and check/update:
- `paths.train_csv` / `paths.valid_csv` / `paths.data_dirs` — your data split CSVs and H5 directories
- `data.tabular_csv` — the clinical features CSV
- `model_weights.model_ckpt` — a pretrained baseline ResNet3D checkpoint (no clinical fusion) that the frozen imaging backbone initializes from. All three architectures freeze most or all of this backbone in `on_train_start`, so this checkpoint matters — training from a random backbone will not produce a meaningful model.
- `model_weights.clinical_ckpt` — optional; a clinical-MLP checkpoint produced by `mlpclinical.py`. If left `null`, the frozen clinical encoder uses random (frozen) weights instead of a pretrained one.
- `model_weights.save_weights_dir` / `debugging.preds_dir` — where checkpoints and (optionally) debug predictions get written

`train.py` trains with early stopping + checkpointing on `best_val_pirads_auc`, then runs `trainer.test()` on `paths.test_csv` if provided, otherwise on the validation set.

## What this doesn't cover

- The older `lakshita_earlyfusion_train.py` script and its YAML
  (`train_fusion_multihead_v2.yaml`) are a separate, earlier pipeline that
  depends on modules not included here (`src/data/lakshita_loader.py`,
  `src/utils/configs/config_enums.py`, `src/models/ResNet3D/lakshita_earlyfusion.py`).
  It predates the three final architectures above and isn't needed to run them.
- The tabular-only baseline (`Tabulartrainsplit.py`, `fintab.py`) and the
  hybrid XGBoost/SHAP comparison scripts (`run_hybrid_comparison.py` and
  friends) are independent of the imaging fusion pipeline and don't go
  through `train.py`.
- Every other `july*`-prefixed file is a historical snapshot kept for
  reference, not a maintained, runnable experiment.
