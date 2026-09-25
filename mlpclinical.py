import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet


class ClinicalMLPModel(Base3DResNet):

    def __init__(self, config):
        super().__init__(config)

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.mlp = nn.Sequential(
            nn.Linear(37, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2),
        )

    def forward(self, data_dict, tabular_features):
        return self.mlp(tabular_features)

    def training_step(self, batch, batch_idx):
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.criterion(logits, target)
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            if self.log_configs["log_run"]:
                self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            print("Test Loss", loss)
        return {"test_loss": loss}