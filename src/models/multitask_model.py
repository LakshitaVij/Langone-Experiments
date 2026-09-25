from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import numpy as np
from src.metrics.metrics import epoch_end_metrics


class MultiTaskModel(pl.LightningModule):
    def __init__(self, config, base_model):
        super().__init__()

        self.save_hyperparameters("config")
        self.config = config
        self._set_default_config()
        self._set_config(config)
        # Mapping from accession number -> dictionary of volumes/heatmaps to save
        self.test_volumes_to_save = {}
        self.max_test_volumes = self.config.get("debugging", {}).get(
            "max_test_volumes", -1
        )
        self.compute_gradcam = self.config.get("debugging", {}).get(
            "compute_gradcam", False
        )
        self.base_model = base_model
        self.base_model.fc = nn.Sequential(*list(self.base_model.fc.children())[:-1])

        self.loss_weight_pirads = config["training"]["mt_loss_weight_pirads"]
        self.loss_weight_gleason = config["training"]["mt_loss_weight_gleason"]
        # self.loss_weight_pirads = nn.Parameter(torch.tensor(1.0))
        # self.loss_weight_gleason = nn.Parameter(torch.tensor(1.0))

        self.pirads_head = nn.Linear(256, 2)
        self.gleason_head = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])

    def forward(self, data_dict):
        x = self.base_model(data_dict)

        out_pirads = self.pirads_head(self.dropout(x))
        out_gleason = self.gleason_head(self.dropout(x))

        return out_pirads, out_gleason

    def _generate_gradcam(
        self, sample_dict, task="pirads", branch="t2", class_idx=None
    ):
        """Compute a simple Grad-CAM heatmap for a single sample.

        Parameters
        ----------
        sample_dict : dict
            A dictionary containing a single sample in the same format as the
            forward method expects.
        task : str
            Which task head to use when computing the Grad-CAM score. Must be
            either ``"pirads"`` or ``"gleason"``.
        branch : str, optional
            Network branch for which to compute the heatmap. Accepted values are
            ``"t2"`` and ``"dwi"``. The values ``"adc"`` and ``"b1500"`` are
            treated as aliases for ``"dwi"`` when the two DWI series are
            stacked as model input.
        class_idx : int or Tensor, optional
            Class index for which to compute the Grad-CAM heatmap. If ``None``,
            uses the predicted class (``argmax`` of the chosen head's logits).
        """
        branch = branch.lower()
        branch = "dwi" if branch in {"dwi", "adc", "b1500"} else branch

        def resolve_target_layer():
            base = self.base_model
            if hasattr(base, "branches"):
                branches = base.branches
                if branch in {"t2", "axt2"}:
                    for name in ("axt2", "t2"):
                        if name in branches:
                            return branches[name].layer2
                elif branch == "dwi":
                    if getattr(base, "stack_adc_b1500", False) and "adc_b1500" in branches:
                        return branches["adc_b1500"].layer2
                    for name in ("adc", "b1500", "dwi"):
                        if name in branches:
                            return branches[name].layer2
                elif branch in branches:
                    return branches[branch].layer2

            if branch in {"t2", "axt2"} and hasattr(base, "resnet_single_branch"):
                return base.resnet_single_branch.layer2
            if branch == "dwi" and hasattr(base, "resnet_dual_branch1"):
                return base.resnet_dual_branch1.layer2
            return None

        target_layer = resolve_target_layer()
        if target_layer is None:
            raise ValueError("branch must be 't2', 'dwi', or a known model branch")

        activations = []
        gradients = []

        def f_hook(_module, _input, output):
            activations.append(output)

        def b_hook(_module, grad_input, grad_output):
            gradients.append(grad_output[0])

        h1 = target_layer.register_forward_hook(f_hook)
        h2 = target_layer.register_full_backward_hook(b_hook)

        # Ensure gradients are tracked even when called from within a
        # ``torch.no_grad()`` or ``torch.inference_mode()`` context (e.g. during
        # evaluation). Temporarily disable inference mode and re-enable gradient
        # calculation so that Grad-CAM can compute gradients properly.
        with torch.inference_mode(False), torch.enable_grad():
            # Inputs passed in from the evaluation loop may have been created
            # under ``torch.inference_mode`` and thus are incompatible with
            # autograd. Clone the sample tensors to obtain normal tensors that
            # support gradient computation before running the forward pass.
            sample_dict = {k: v.clone() for k, v in sample_dict.items()}
            out_pirads, out_gleason = self(sample_dict)
            if task == "pirads":
                out = out_pirads
            elif task == "gleason":
                out = out_gleason
            else:
                raise ValueError("task must be 'pirads' or 'gleason'")

            if class_idx is None:
                class_idx_tensor = out.argmax(dim=1)
            else:
                class_idx_tensor = (
                    class_idx
                    if torch.is_tensor(class_idx)
                    else torch.tensor([class_idx], device=out.device)
                )
            score = out[
                torch.arange(out.shape[0], device=out.device), class_idx_tensor
            ].sum()
            self.zero_grad()
            score.backward()

        h1.remove()
        h2.remove()

        grad = gradients[0]
        act = activations[0]

        grad_squared = grad ** 2
        grad_cubed = grad ** 3
        denom = 2 * grad_squared + act * grad_cubed + 1e-7
        alpha = torch.where(denom != 0, grad_squared / denom, torch.zeros_like(grad_squared))
        positive_grad = torch.relu(grad)
        weights = (alpha * positive_grad).sum(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))

        if branch in {"t2", "axt2"}:
            input_key = next(
                (k for k in ("axt2", "t2") if k in sample_dict),
                next(iter(sample_dict)),
            )
        elif branch == "dwi":
            input_key = next(
                (k for k in ("dwi", "adc", "b1500") if k in sample_dict),
                next(iter(sample_dict)),
            )
        else:
            input_key = branch if branch in sample_dict else next(iter(sample_dict))

        input_size = sample_dict[input_key].shape[2:]
        cam = F.interpolate(cam, size=input_size, mode="trilinear", align_corners=True)
        return cam[0, 0].detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hyperparams["max_patience"],
            factor=self.hyperparams["factor"],
            threshold=1e-4,
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _set_default_config(self):
        # set default hyperparams
        self.hyperparams = {}
        self.hyperparams["learning_rate"] = 1e-04
        self.hyperparams["weight_decay"] = 0.001
        self.hyperparams["max_patience"] = 5
        self.hyperparams["factor"] = 0.3
        # self.hyperparams["dropout"] = 0.5

        # set logging hyperparams
        self.log_run = False

    def _set_config(self, config):
        self.config = config

        # set the hyperparameters
        for key in self.hyperparams.keys():
            hyperparam_keys = config["hyperparameters"].keys()
            if key in hyperparam_keys:
                self.hyperparams[key] = float(config["hyperparameters"][key])

        # define loss
        self.multitask_sampler = config["training"]["multitask_sampler"]
        reduction = "mean" if self.multitask_sampler else "none"

        if config["training"]["imbalance_strategy"] == "weighted_loss":
            self.pirads_criterion = nn.CrossEntropyLoss(
                weight=config["training"]["class_weights"], reduction=reduction
            )
        else:
            self.pirads_criterion = nn.CrossEntropyLoss(reduction=reduction)

        self.gleason_criterion = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=-1
        )

        self.unweighted_loss_p = nn.CrossEntropyLoss()
        self.unweighted_loss_gs = nn.CrossEntropyLoss(ignore_index=-1)

        # save predictions for epoch level metrics
        self.val_preds = {
            "pirads_preds": [],
            "gleason_preds": [],
            "pirads_targets": [],
            "gleason_targets": [],
            "maxPIRADS": [],
            "MaxGleasonScore": [],
            "AccessionNumber": [],
        }
        self.train_preds = {
            "pirads_preds": [],
            "gleason_preds": [],
            "pirads_targets": [],
            "gleason_targets": [],
        }

        # save best metric across epochs
        self.best_val_pirads_auc = 0
        self.best_val_gleason_auc = 0
        self.best_val_pirads_gleason_auc = 0

        # log hyperparams
        self.log_configs = config["logging"]

    def training_step(self, batch, batch_idx):
        data_dict = batch["volume_data_dict"]
        target_pirads = batch["label"]
        target_gleason = batch["gleason_label"]

        out_pirads, out_gleason = self(data_dict)
        loss_pirads = self.pirads_criterion(out_pirads, target_pirads)
        loss_gleason = self.gleason_criterion(
            out_gleason, target_gleason
        )  # returns zero for cases with ignore_index

        combined_loss = None

        if self.multitask_sampler:
            combined_loss = (
                loss_pirads
                if torch.all(target_gleason == -1)
                else 0.5 * loss_pirads + 0.5 * loss_gleason
            )
        else:
            combined_loss = torch.zeros(len(loss_pirads))
            mask = target_gleason != -1
            combined_loss[mask] = (
                self.loss_weight_pirads * loss_pirads[mask]
                + self.loss_weight_gleason * loss_gleason[mask]
            )
            combined_loss[~mask] = loss_pirads[
                ~mask
            ]  # use only pirads loss for cases with ignore_index

            combined_loss = combined_loss.mean()

        # Save predictions and targets
        self.train_preds["pirads_preds"].append(out_pirads)
        self.train_preds["gleason_preds"].append(out_gleason)
        self.train_preds["pirads_targets"].append(target_pirads)
        self.train_preds["gleason_targets"].append(target_gleason)

        # log loss
        if self.log_configs["log_run"]:
            self.log(
                "train_pirads_loss",
                loss_pirads.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train_gleason_loss",
                loss_gleason.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train_loss", combined_loss, prog_bar=True, on_step=True, on_epoch=True
            )

        return combined_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch["volume_data_dict"]
        target_pirads = batch["label"]
        target_gleason = batch["gleason_label"]

        out_pirads, out_gleason = self(data_dict)
        loss_pirads = self.unweighted_loss_p(out_pirads, target_pirads)
        loss_gleason = self.unweighted_loss_gs(out_gleason, target_gleason)

        combined_loss = loss_pirads + loss_gleason

        self.val_preds["pirads_preds"].append(out_pirads)
        self.val_preds["gleason_preds"].append(out_gleason)
        self.val_preds["pirads_targets"].append(target_pirads)
        self.val_preds["gleason_targets"].append(target_gleason)
        self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
        self.val_preds["MaxGleasonScore"].append(batch["MaxGleasonScore"])
        self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])

        if self.log_configs["log_run"]:
            self.log(
                "val_pirads_loss",
                loss_pirads,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "val_gleason_loss",
                loss_gleason,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "val_loss", combined_loss, prog_bar=True, on_step=True, on_epoch=True
            )

        return {"val_loss": combined_loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Force dropout to be in train mode when we are doing monte carlo inference

        # for module in self.modules():
        #    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        #        module.train()

        data_dict = batch["volume_data_dict"]
        unnorm_data_dict = batch.get("unnorm_volume_data_dict")

        if not hasattr(self, "saved_volumes"):
            self.saved_volumes = False

        # Optionally accumulate a subset of volumes in the test set
        if self.max_test_volumes != 0:
            batch_size = len(batch["AccessionNumber"])
            for i in range(batch_size):
                if (
                    self.max_test_volumes > 0
                    and len(self.test_volumes_to_save) >= self.max_test_volumes
                ):
                    break
                acc = f"{batch['AccessionNumber'][i].item()}"
                case_dict = self.test_volumes_to_save.setdefault(acc, {})
                for key in data_dict:
                    vols = (
                        unnorm_data_dict.get(key, data_dict[key])
                        if unnorm_data_dict is not None
                        else data_dict[key]
                    )
                    case_dict[key] = vols[i, 0].detach().cpu().numpy()

                if self.compute_gradcam:
                    sample_dict = {k: v[i : i + 1] for k, v in data_dict.items()}
                    case_dict["gradcam_pirads_t2"] = self._generate_gradcam(
                        sample_dict, task="pirads", branch="t2"
                    )
                    case_dict["gradcam_gleason_t2"] = self._generate_gradcam(
                        sample_dict, task="gleason", branch="t2"
                    )
                    case_dict["gradcam_pirads_dwi"] = self._generate_gradcam(
                        sample_dict, task="pirads", branch="dwi"
                    )
                    case_dict["gradcam_gleason_dwi"] = self._generate_gradcam(
                        sample_dict, task="gleason", branch="dwi"
                    )

        target_pirads = batch["label"]
        target_gleason = batch["gleason_label"]

        out_pirads, out_gleason = self(data_dict)
        loss_pirads = self.unweighted_loss_p(out_pirads, target_pirads)
        loss_gleason = self.unweighted_loss_gs(out_gleason, target_gleason)

        combined_loss = loss_pirads + loss_gleason

        self.val_preds["pirads_preds"].append(out_pirads)
        self.val_preds["gleason_preds"].append(out_gleason)
        self.val_preds["pirads_targets"].append(target_pirads)
        self.val_preds["gleason_targets"].append(target_gleason)
        self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
        self.val_preds["MaxGleasonScore"].append(batch["MaxGleasonScore"])
        self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])

        if self.log_configs["log_run"]:
            self.log(
                "test_pirads_loss",
                loss_pirads,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "test_gleason_loss",
                loss_gleason,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "test_loss", combined_loss, prog_bar=True, on_step=True, on_epoch=True
            )

        return {"test_loss": combined_loss}

    def on_train_epoch_end(self):
        if self.log_configs["log_run"]:
            # Convert list of tensors to single tensor for PIRADS
            train_pirads_preds = torch.cat(self.train_preds["pirads_preds"], dim=0)
            train_pirads_targets = torch.cat(self.train_preds["pirads_targets"], dim=0)

            # Convert list of tensors to single tensor for Gleason
            train_gleason_preds = torch.cat(self.train_preds["gleason_preds"], dim=0)
            train_gleason_targets = torch.cat(
                self.train_preds["gleason_targets"], dim=0
            )

            valid_indices = train_gleason_targets != -1
            train_gleason_preds = train_gleason_preds[valid_indices]
            train_gleason_targets = train_gleason_targets[valid_indices]

            # Compute per epoch stats
            (
                pirads_auc,
                pirads_opt_threshold,
                pirads_precision,
                pirads_recall,
                pirads_f1,
            ) = epoch_end_metrics(
                train_pirads_preds, train_pirads_targets, self.current_epoch
            )
            (
                gleason_auc,
                gleason_opt_threshold,
                gleason_precision,
                gleason_recall,
                gleason_f1,
            ) = epoch_end_metrics(
                train_gleason_preds, train_gleason_targets, self.current_epoch
            )

            self.log("train_pirads_auc", pirads_auc, prog_bar=True, on_epoch=True)
            self.log("train_gleason_auc", gleason_auc, prog_bar=True, on_epoch=True)

        self.train_preds = {
            "pirads_preds": [],
            "gleason_preds": [],
            "pirads_targets": [],
            "gleason_targets": [],
        }

    def on_validation_epoch_end(self):
        if self.log_configs["log_run"]:
            # Convert list of tensors to single tensor for PIRADS
            val_pirads_preds = torch.cat(self.val_preds["pirads_preds"], dim=0)
            val_pirads_targets = torch.cat(self.val_preds["pirads_targets"], dim=0)

            # Convert list of tensors to single tensor for Gleason
            val_gleason_preds_all = torch.cat(self.val_preds["gleason_preds"], dim=0)
            val_gleason_targets_all = torch.cat(
                self.val_preds["gleason_targets"], dim=0
            )

            # Treat ``-1`` labels as negative for metric calculation
            val_gleason_targets = val_gleason_targets_all.clone()
            val_gleason_targets[val_gleason_targets == -1] = 0
            val_gleason_preds = val_gleason_preds_all

            # Track metadata
            val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)
            val_max_gleason = torch.cat(self.val_preds["MaxGleasonScore"], dim=0)
            val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)

            # Compute per epoch stats
            (
                pirads_auc,
                pirads_opt_threshold,
                pirads_precision,
                pirads_recall,
                pirads_f1,
            ) = epoch_end_metrics(
                val_pirads_preds,
                val_pirads_targets,
                self.current_epoch,
                plot_roc=self.log_configs["plot_roc"],
                plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
                plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
                pirads=self.log_configs["val_pirads"],
                mode="val",
            )

            (
                gleason_auc,
                gleason_opt_threshold,
                gleason_precision,
                gleason_recall,
                gleason_f1,
            ) = epoch_end_metrics(
                val_gleason_preds, val_gleason_targets, self.current_epoch, mode="val"
            )

            # Compute pirads predictions against gleason target
            (
                pirads_gleason_auc,
                pirads_gleason_opt_threshold,
                pirads_gleason_precision,
                pirads_gleason_recall,
                pirads_gleason_f1,
            ) = epoch_end_metrics(
                val_pirads_preds,
                val_gleason_targets,
                self.current_epoch,
                mode="val",
            )

            if pirads_gleason_auc > self.best_val_pirads_gleason_auc:
                self.best_val_pirads_gleason_auc = pirads_gleason_auc
                self.best_val_gleason_auc = gleason_auc
                self.best_val_pirads_auc = pirads_auc

            # Log PIRADS metrics
            self.log("val_pirads_auc", pirads_auc, prog_bar=True, on_epoch=True)
            self.log(
                "val_pirads_precision", pirads_precision, prog_bar=True, on_epoch=True
            )
            self.log("val_pirads_recall", pirads_recall, prog_bar=True, on_epoch=True)
            self.log(
                "best_val_pirads_auc",
                self.best_val_pirads_auc,
                prog_bar=True,
                on_epoch=True,
            )

            self.log("val_gleason_auc", gleason_auc, prog_bar=True, on_epoch=True)
            self.log(
                "best_val_gleason_auc",
                self.best_val_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )

            self.log(
                "val_pirads_gleason_auc",
                pirads_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )
            self.log(
                "best_val_pirads_gleason_auc",
                self.best_val_pirads_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )

            if self.config["debugging"]["debug"]:
                # save predictions for debugging
                softmax = Softmax(dim=1)
                data_dict = {
                    "AccessionNumber": val_acc_num,
                    "val_pirads_preds": softmax(val_pirads_preds),
                    "val_pirads_targets": val_pirads_targets,
                    "maxPIRADS": val_max_pirads,
                    "val_gleason_preds": softmax(val_gleason_preds_all),
                    "val_gleason_targets": val_gleason_targets_all,
                    "MaxGleasonScore": val_max_gleason,
                }
                for k, v in data_dict.items():
                    data_dict[k] = v.cpu().tolist()

                df = pd.DataFrame(data_dict)
                file_name = f"preds_epoch_{self.current_epoch}_pirads_auc_{pirads_auc}_best_auc_{self.best_val_pirads_auc}.csv"
                df.to_csv(
                    Path(self.config["debugging"]["preds_dir"]) / file_name, index=False
                )

            self.val_preds = {
                "pirads_preds": [],
                "gleason_preds": [],
                "pirads_targets": [],
                "gleason_targets": [],
                "maxPIRADS": [],
                "MaxGleasonScore": [],
                "AccessionNumber": [],
            }

    def on_test_epoch_end(self):
        if self.log_configs["log_run"]:
            # Convert list of tensors to single tensor for PIRADS
            val_pirads_preds = torch.cat(self.val_preds["pirads_preds"], dim=0)
            val_pirads_targets = torch.cat(self.val_preds["pirads_targets"], dim=0)

            # Convert list of tensors to single tensor for Gleason
            val_gleason_preds_all = torch.cat(self.val_preds["gleason_preds"], dim=0)
            val_gleason_targets_all = torch.cat(
                self.val_preds["gleason_targets"], dim=0
            )

            # Treat ``-1`` labels as negative for metric calculation
            val_gleason_targets = val_gleason_targets_all.clone()
            val_gleason_targets[val_gleason_targets == -1] = 0
            val_gleason_preds = val_gleason_preds_all

            # Track metadata
            val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)
            val_max_gleason = torch.cat(self.val_preds["MaxGleasonScore"], dim=0)
            val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)

            # Compute per epoch stats
            (
                pirads_auc,
                pirads_opt_threshold,
                pirads_precision,
                pirads_recall,
                pirads_f1,
            ) = epoch_end_metrics(
                val_pirads_preds,
                val_pirads_targets,
                self.current_epoch,
                plot_roc=self.log_configs["plot_roc"],
                plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
                plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
                pirads=self.log_configs["val_pirads"],
                mode="val",
            )

            (
                gleason_auc,
                gleason_opt_threshold,
                gleason_precision,
                gleason_recall,
                gleason_f1,
            ) = epoch_end_metrics(
                val_gleason_preds, val_gleason_targets, self.current_epoch, mode="val"
            )

            # Compute pirads predictions against gleason target
            (
                pirads_gleason_auc,
                pirads_gleason_opt_threshold,
                pirads_gleason_precision,
                pirads_gleason_recall,
                pirads_gleason_f1,
            ) = epoch_end_metrics(
                val_pirads_preds,
                val_gleason_targets,
                self.current_epoch,
                mode="val",
            )

            if pirads_gleason_auc > self.best_val_pirads_gleason_auc:
                self.best_val_pirads_gleason_auc = pirads_gleason_auc
                self.best_val_gleason_auc = gleason_auc
                self.best_val_pirads_auc = pirads_auc

            # Log PIRADS metrics
            self.log("test_pirads_auc", pirads_auc, prog_bar=True, on_epoch=True)
            self.log(
                "test_pirads_precision", pirads_precision, prog_bar=True, on_epoch=True
            )
            self.log("test_pirads_recall", pirads_recall, prog_bar=True, on_epoch=True)
            self.log(
                "best_test_pirads_auc",
                self.best_val_pirads_auc,
                prog_bar=True,
                on_epoch=True,
            )

            self.log("test_gleason_auc", gleason_auc, prog_bar=True, on_epoch=True)
            self.log(
                "best_test_gleason_auc",
                self.best_val_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )

            self.log(
                "test_pirads_gleason_auc",
                pirads_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )
            self.log(
                "best_test_pirads_gleason_auc",
                self.best_val_pirads_gleason_auc,
                prog_bar=True,
                on_epoch=True,
            )

            # Save all test volumes after final batch
            save_dir = (
                Path(self.config["debugging"]["preds_dir"])
                / f"volumes_epoch_{self.current_epoch}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            for acc, data_dict in self.test_volumes_to_save.items():
                np.savez(save_dir / f"{acc}.npz", **data_dict)

            self.test_volumes_to_save.clear()

            if self.config["debugging"]["debug"]:
                # save predictions for debugging
                softmax = Softmax(dim=1)
                data_dict = {
                    "AccessionNumber": val_acc_num,
                    "val_pirads_preds": softmax(val_pirads_preds),
                    "val_pirads_targets": val_pirads_targets,
                    "maxPIRADS": val_max_pirads,
                    "val_gleason_preds": softmax(val_gleason_preds_all),
                    "val_gleason_targets": val_gleason_targets_all,
                    "MaxGleasonScore": val_max_gleason,
                }
                for k, v in data_dict.items():
                    data_dict[k] = v.cpu().tolist()

                df = pd.DataFrame(data_dict)
                # <<<<< NEW: grab run_num if it exists, default=0 >>>>>
                run_num = getattr(self, "run_num", None)
                save_name = getattr(self, "save_name", "preds")
                model_label = getattr(self, "model_label", "model")

                # Build a concise file name that captures model type, inference
                # setting and resulting AUC.  Include the run number if one is
                # provided.
                file_name = f"{model_label}_{save_name}_auc_{pirads_auc:.3f}"
                if run_num is not None:
                    file_name += f"_run_{run_num}"
                file_name += ".csv"

                # Then save the CSV
                df.to_csv(
                    Path(self.config["debugging"]["preds_dir"]) / file_name, index=False
                )

            self.val_preds = {
                "pirads_preds": [],
                "gleason_preds": [],
                "pirads_targets": [],
                "gleason_targets": [],
                "maxPIRADS": [],
                "MaxGleasonScore": [],
                "AccessionNumber": [],
            }
