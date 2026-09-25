# Multimodal csPCa Detection

Detecting clinically significant prostate cancer (csPCa) by fusing 3D MRI
(T2 / ADC / B1500) with structured clinical features (PSA, PI-RADS, lesion
location, etc.) using a frozen ResNet3D imaging backbone.

## Architectures

| Architecture | Model | Config | Writeup |
|---|---|---|---|
| CBAM (channel + spatial clinical attention) | `aug11cbam.py` | `configs/cbam.yaml` | `cbam.md` |
| Early-scalar clinical attention | `aug11earlyscalar.py` | `configs/earlyscalar.yaml` | `earlyscalar.md` |
| Late fusion (flat clinical encoder) | `aug11latefusionflat.py` | `configs/latefusion_flat.yaml` | `latefusion.md` |

Each `.md` file (with matching `.png` diagram) explains the design. This
README only covers how to run them.

## Setup

```bash
pip install -r requirements.txt
```

This runs on the lab's HPC cluster (`prostatelab`) — the H5 imaging data and
CSV splits live under `/gpfs/data/prostatelab/...`. You'll need access to
that data (or an equivalent copy) for any of this to do anything.

## Data format

Each exam is one H5 file with:
```
['axt2'], ['adc'], ['b1500']        # imaging volumes
.attrs['maxPIRADS'], .attrs['psa'], .attrs['prostate_volume'], ...
```
Clinical features come from a separate CSV, keyed by `AccessionNumber`, with
37 columns matching what the clinical encoder expects.

## Running training

```bash
python train.py --config configs/cbam.yaml
python train.py --config configs/earlyscalar.yaml
python train.py --config configs/latefusion_flat.yaml
```

Before running, check each config for:
- `paths.train_csv` / `paths.valid_csv` / `paths.data_dirs`
- `data.tabular_csv`
- `model_weights.model_ckpt` — a pretrained baseline checkpoint the frozen
  backbone initializes from (training freezes most/all of it in
  `on_train_start`, so this matters)
- `model_weights.clinical_ckpt` — optional pretrained clinical MLP checkpoint

Training uses early stopping + checkpointing on `best_val_pirads_auc`.

## `src/`

The `src/` package (data loading, ResNet3D backbone, metrics) mirrors the
lab's shared module of the same name, included here so training actually
runs from this repo rather than assuming everyone already has it on their
`PYTHONPATH`.
