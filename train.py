"""
Config-driven training entrypoint for the three final frozen-encoder fusion
architectures:

    cbam            -> aug11cbam.TriSeriesModel
    earlyscalar     -> aug11earlyscalar.TriSeriesModelFrozen
    latefusion_flat -> aug11latefusionflat.LateFusionFlatFrozen

All three subclass src.models.ResNet3D.base_3Dresnet.Base3DResNet and expect a
frozen imaging backbone (loaded from model_weights.model_ckpt) plus an
optional frozen clinical MLP encoder (model_weights.clinical_ckpt).

Usage:
    python train.py --config configs/cbam.yaml
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, default_collate

from src.data.loader import ExamH5Dataset
from src.models.ResNet3D.lakshita_earlyfusion_construct_resnet3d import (
    load_saved_resnet3d_weights,
)
from src.utils.data_enums import SeriesType

from aug11cbam import TriSeriesModel as CBAMModel
from aug11earlyscalar import TriSeriesModelFrozen as EarlyScalarModel
from aug11latefusionflat import LateFusionFlatFrozen as LateFusionFlatModel


MODEL_REGISTRY = {
    "cbam": CBAMModel,
    "earlyscalar": EarlyScalarModel,
    "latefusion_flat": LateFusionFlatModel,
}

SERIES_NAME_TO_ENUM = {
    "axt2": SeriesType.AXT2,
    "adc": SeriesType.ADC,
    "b1500": SeriesType.B1500,
    "dce": SeriesType.DCE,
}

TARGET_FROM_MODE = {
    "pirads": "pirads",
    "gleason": "gleason",
    "tstage": "tstage",
    "cspca": "cspca",
}


def collate_with_tabular_alias(batch):
    """ExamH5Dataset returns the clinical feature tensor as 'TabularFeatures',
    but the aug11 fusion models' training/validation/test steps read
    batch['tabular_features']. Alias it here instead of editing either the
    recovered loader or the recovered model files."""
    collated = default_collate(batch)
    collated["tabular_features"] = collated["TabularFeatures"]
    return collated


def build_dataset(csv_path, config, mode, series):
    data_cfg = config["data"]
    return ExamH5Dataset(
        metadata_csv=csv_path,
        data_dirs=config["paths"]["data_dirs"],
        series=series,
        model_type="3D",
        augment=data_cfg.get("augment", "none") if mode == "train" else "none",
        noise_sigma_range=data_cfg.get("noise_sigma_range", (0.0, 0.15)),
        downsample_factors=data_cfg.get("downsample_factors"),
        pirads_cutoff=data_cfg["pirads_cutoff"],
        mask_prostate=data_cfg.get("mask_prostate", False),
        device="cpu",
        mode=mode,
        normalize=data_cfg.get("normalize", True),
        target=TARGET_FROM_MODE[config["training"].get("mode", "pirads")],
        axt2_key=data_cfg.get("axt2_key", "axt2"),
        dwi_suffices=data_cfg.get("dwi_suffices"),
        dce_dirs=config["paths"].get("dce_dirs"),
        tabular_csv=data_cfg.get("tabular_csv"),
    )


def build_dataloader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        shuffle=shuffle,
        collate_fn=collate_with_tabular_alias,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config):
    model_name = config["model"]["model"].lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of {list(MODEL_REGISTRY)}."
        )
    model = MODEL_REGISTRY[model_name](config)

    weights_cfg = config.get("model_weights", {})
    if weights_cfg.get("load_weights") and weights_cfg.get("model_ckpt"):
        checkpoint = torch.load(
            weights_cfg["model_ckpt"],
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # Non-matching keys (the attention/clinical-fusion layers, which the
        # checkpoint - trained on the plain baseline branched model - doesn't
        # have) are silently skipped, leaving them at their random init.
        model = load_saved_resnet3d_weights(
            model,
            checkpoint,
            disable_gradient=weights_cfg.get("disable_gradient", False),
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train one of the final fusion architectures (cbam / earlyscalar / latefusion_flat)"
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("paths", {}).setdefault("extra_valid_csv", None)
    config.setdefault("paths", {}).setdefault("dce_dirs", [])

    torch.manual_seed(config["training"].get("seed", 42))

    series = [SERIES_NAME_TO_ENUM[s] for s in config["data"]["series"]]

    train_dataset = build_dataset(config["paths"]["train_csv"], config, "train", series)
    val_dataset = build_dataset(config["paths"]["valid_csv"], config, "val", series)

    if config["training"].get("imbalance_strategy") == "weighted_loss" and not config[
        "training"
    ].get("class_weights"):
        config["training"]["class_weights"] = train_dataset.class_weights

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    config["logging"]["val_pirads"] = [val_dataset.pirads]

    model = build_model(config)

    use_gpu = config["training"].get("gpu", False) and torch.cuda.is_available()
    if use_gpu:
        model.to(torch.device("cuda:0"))

    save_dir = Path(config["model_weights"]["save_weights_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if config["debugging"].get("debug"):
        Path(config["debugging"]["preds_dir"]).mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="best_val_pirads_auc",
            patience=int(config["hyperparameters"].get("max_patience", 20)),
            mode="max",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="best_val_pirads_auc",
            dirpath=save_dir,
            filename=config["model"]["model"] + "_{epoch:02d}_{best_val_pirads_auc:.3f}",
            mode="max",
            save_top_k=1,
        ),
    ]

    trainer_kwargs = dict(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=callbacks,
        max_epochs=int(config["training"]["epochs"]),
        default_root_dir=save_dir,
        num_sanity_val_steps=0,
    )
    if config["logging"].get("log_run"):
        trainer_kwargs["logger"] = WandbLogger(
            name=config["logging"]["run_name"],
            project=config["logging"]["project_name"],
            save_dir=str(save_dir),
        )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_dataloaders=[val_loader])

    test_csv = config["paths"].get("test_csv")
    if test_csv:
        test_dataset = build_dataset(test_csv, config, "test", series)
        test_loader = build_dataloader(test_dataset, config, shuffle=False)
    else:
        test_loader = val_loader
    trainer.test(model, dataloaders=[test_loader])


if __name__ == "__main__":
    main()
