"""
FOR LAX REF:
- This is the changed file , this is used as the actual training for early fusion, we sub things in and out of here
- THE MAIN change here out of everything , is that this actually loads the tab csv brother

- 
lakshita_earlyfusion_train.py = the customer. Just says "give me a model"
lakshita_earlyfusion_construction.py = the waiter. Takes the order, figures out what to build
cbam.py = the kitchen. Actually builds the model
"""


import functools
import os
import random
import sys
from pathlib import Path
import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import yaml
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data.lakshita_loader import load_data
from src.models.multitask_model import MultiTaskModel
from src.models.ResNet3D.lakshita_earlyfusion_construction  import generate_resnet3d
from src.utils.configs.config_enums import ValidTrainConfigs


def train(config, ckpt_path=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    # if we're running a hyperparameter sweep, use wandb config
    if config["hyperparameters"]["sweep"]:
        wandb.init(project=config["logging"]["project_name"])
        wandb_config = wandb.config
        for key in wandb_config.keys():
            if key in ValidTrainConfigs.TRAINING.value:
                config["training"][key] = wandb_config[key]
            elif key in ValidTrainConfigs.MRNET_MODEL.value:
                config["model"][key] = wandb_config[key]
            elif key in ValidTrainConfigs.HYPERPARAMETERS.value:
                config["hyperparameters"][key] = wandb_config[key]
            elif key == "series":
                config["model"][key] = wandb_config[key]

    # validate configs
    config = ValidTrainConfigs.validate_train_configs(config)

    train_csv = config["paths"]["train_csv"]
    valid_csv = config["paths"]["valid_csv"]
    data_dirs = config["paths"]["data_dirs"]
    dce_dirs = config["paths"].get("dce_dirs", [])
    pirads_cutoff = int(config["data"]["pirads_cutoff"])
    mode = config["training"].get("mode", "pirads")
    target_type = "pirads"
    if mode == "gleason":
        target_type = "gleason"
    elif mode == "tstage":
        target_type = "tstage"
    elif mode == "cspca":
        target_type = "cspca"

    # set device and seed for reproducibility
    seed = int(config["training"]["seed"])
    gpu = config["training"]["gpu"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # set model_type
    model_type = (
        "3D" if config["model"]["model"].lower() in ["resnet3d", "vit3d"] else "2D"
    )

    # set batch size and gradient accumulation
    batch_size = config["training"]["batch_size"]
    grad_acc = batch_size // 4
    loader_target = "pirads" if mode == "multitask" else target_type

    # create data loaders
    if config["training"]["imbalance_strategy"] == "weighted_sampling":
        train_loader = load_data(
            train_csv,
            data_dirs,
            dce_dirs=dce_dirs,
            batch_size=batch_size,
            series=config["data"]["series"],
            model_type=model_type,
            augment=config["data"]["augment"],
            noise_sigma_range=config["data"]["noise_sigma_range"],
            downsample_factors=config["data"]["downsample_factors"],
            pirads_cutoff=pirads_cutoff,
            mask_prostate=config["data"]["mask_prostate"],
            device=device,
            shuffle=False,  # sampler handles shuffling
            weighted_sample=True,  # address class imbalance while training
            mode="train",
            normalize=config["data"].get("normalize", True),
            target=loader_target,
            axt2_key=config["data"].get("axt2_key", "axt2"),
            dwi_suffices=config["data"].get("dwi_suffices"),
            tabular_csv=config["data"].get("tabular_csv")
        )
    elif mode == "multitask":
        if config["training"]["multitask_sampler"]:
            train_loader = load_data(
                train_csv,
                data_dirs,
                dce_dirs=dce_dirs,
                batch_size=batch_size,
                series=config["data"]["series"],
                model_type=model_type,
                augment=config["data"]["augment"],
                noise_sigma_range=config["data"]["noise_sigma_range"],
                downsample_factors=config["data"]["downsample_factors"],
                pirads_cutoff=pirads_cutoff,
                mask_prostate=config["data"]["mask_prostate"],
                device=device,
                shuffle=True,
                weighted_sample=False,
                multitask_sampler=config["training"]["multitask_sampler"],
                mode="train",
                normalize=config["data"].get("normalize", True),
                target=loader_target,
                axt2_key=config["data"].get("axt2_key", "axt2"),
                dwi_suffices=config["data"].get("dwi_suffices"),
            )
    else:
        train_loader = load_data(
            train_csv,
            data_dirs,
            dce_dirs=dce_dirs,
            batch_size=batch_size,
            series=config["data"]["series"],
            model_type=model_type,
            augment=config["data"]["augment"],
            noise_sigma_range=config["data"]["noise_sigma_range"],
            downsample_factors=config["data"]["downsample_factors"],
            pirads_cutoff=pirads_cutoff,
            mask_prostate=config["data"]["mask_prostate"],
            device=device,
            shuffle=True,
            weighted_sample=False,
            mode="train",
            normalize=config["data"].get("normalize", True),
            target=loader_target,
            axt2_key=config["data"].get("axt2_key", "axt2"),
            dwi_suffices=config["data"].get("dwi_suffices"),
            tabular_csv=config["data"].get("tabular_csv")
        )
    if config["training"]["imbalance_strategy"] == "weighted_loss":
        if config["training"]["class_weights"] is None:
            config["training"]["class_weights"] = train_loader.dataset.class_weights

    print("Config settings:\n", config)

    # validation loader for the full val dataset
    val_loaders = []
    config["logging"]["val_pirads"] = []
    full_valid_loader = load_data(
        valid_csv,
        data_dirs,
        dce_dirs=dce_dirs,
        batch_size=batch_size,
        series=config["data"]["series"],
        model_type=model_type,
        augment="none",  # dont augment validation data
        noise_sigma_range=config["data"]["noise_sigma_range"],
        downsample_factors=config["data"]["downsample_factors"],
        pirads_cutoff=pirads_cutoff,
        mask_prostate=config["data"]["mask_prostate"],
        device=device,
        shuffle=False,  # do not shuffle validation data
        weighted_sample=False,  # do not address class imbalance while validating
        mode="val",
        normalize=config["data"].get("normalize", True),
        target=loader_target,
        axt2_key=config["data"].get("axt2_key", "axt2"),
        dwi_suffices=config["data"].get("dwi_suffices"),
        tabular_csv=config["data"].get("tabular_csv")
    )
    val_loaders.append(full_valid_loader)
    config["logging"]["val_pirads"].append(full_valid_loader.dataset.pirads)

    # validation loader for an extra validation set
    if config["paths"]["extra_valid_csv"] is not None:
        extra_valid_loader = load_data(
            config["paths"]["extra_valid_csv"],
            data_dirs,
            dce_dirs=dce_dirs,
            batch_size=batch_size,
            series=config["model"]["series"],
            model_type=model_type,
            augment="none",  # do not augment validation data
            noise_sigma_range=config["data"]["noise_sigma_range"],
            downsample_factors=config["data"]["downsample_factors"],
            pirads_cutoff=pirads_cutoff,
            mask_prostate=config["data"]["mask_prostate"],
            device=device,
            shuffle=False,  # do not shuffle validation data
            weighted_sample=False,  # do not address class imbalance while validating
            mode="val",
            normalize=config["data"].get("normalize", True),
            target=loader_target,
            axt2_key=config["data"].get("axt2_key", "axt2"),
            dwi_suffices=config["data"].get("dwi_suffices"),
        )
        val_loaders.append(extra_valid_loader)
        config["logging"]["val_pirads"].append(extra_valid_loader.dataset.pirads)

    # init the specified model
    if config["model"]["model"].lower() == "resnet3d":
        model = generate_resnet3d(config)
        if mode == "multitask":
            model = MultiTaskModel(config, model)
    elif config["model"]["model"].lower() == "vit3d":
        model = generate_vit3d(config)
    else:
        raise ValueError("Model not recognized.")

    # set args for trainer
    epochs = int(config["training"]["epochs"])
    weights_dir = config["model_weights"]["save_weights_dir"]

    # move model to gpu if available
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)

    # The metric used for early stopping / checkpoint selection differs by
    # model: the multitask model logs a combined PI-RADS+Gleason AUC, while the
    # single-task ResNet3D (pirads / gleason / cspca / tstage) logs
    # `best_val_pirads_auc`. Monitoring the wrong name silently disables
    # checkpointing, so pick the one the active model actually logs.
    monitor_metric = (
        "best_val_pirads_gleason_auc" if mode == "multitask" else "best_val_pirads_auc"
    )

    # configure early stopping
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric, patience=20, mode="max"
    )
    series = [series.value["key"] for series in config["data"]["series"]]
    series_str = "_".join(series)
    augment = config["data"].get("augment", "none")
    augment_str_map = {"none": "baseline", "noise": "noise_augmented", "downsample": "res_augmented"}
    augment_str = augment_str_map.get(augment, augment)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=weights_dir,
        filename=f"{{epoch:02d}}_AUC_{{{monitor_metric}:.2f}}_{series_str}_{augment_str}_{mode}_pirads_cutoff_{pirads_cutoff}",
        mode="max",
        save_top_k=1,
    )

    # Always checkpoint and early-stop; W&B logging is the only thing gated on
    # log_run.
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    trainer_kwargs = dict(
        accelerator="gpu",
        callbacks=callbacks,
        accumulate_grad_batches=grad_acc,
        devices=1,
        max_epochs=epochs,
        default_root_dir=weights_dir,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    if config["logging"]["log_run"]:
        trainer_kwargs["logger"] = WandbLogger(
            name=config["logging"]["run_name"],
            project=config["logging"]["project_name"],
            save_dir=weights_dir,
            offline=False,
        )
    trainer = pl.Trainer(**trainer_kwargs)
    if ckpt_path:
        if Path(ckpt_path).exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    trainer.fit(
        model,
        train_loader,
        val_dataloaders=val_loaders,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="Path to configuration YAML")
    parser.add_argument(
        "sweep_id",
        nargs="?",
        default=None,
        help="Optional Weights & Biases sweep ID",
    )
    parser.add_argument(
        "--augment",
        choices=["none", "standard", "noise", "downsample"],
        help="Override the augmentation strategy defined in the config",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Path to a checkpoint file to resume training",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} does not exist")

    config = yaml.safe_load(Path(args.config).read_text())

    if args.augment is not None:
        config.setdefault("data", {})
        config["data"]["augment"] = args.augment

    if args.sweep_id is not None:
        config["hyperparameters"]["sweep"] = True
    else:
        config["hyperparameters"]["sweep"] = False

    mp.set_start_method("spawn")

    if config["hyperparameters"]["sweep"]:
        train_partial = functools.partial(train, config=config, ckpt_path=args.ckpt_path)
        wandb.agent(sweep_id=args.sweep_id, function=train_partial, count=5)
    else:
        train(config, ckpt_path=args.ckpt_path)