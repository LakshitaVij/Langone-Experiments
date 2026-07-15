import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch


class CrossAttentionFusionModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 3

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)

        # imaging branches
        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.num_branches = 0
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.num_branches += 1
                    self._dwi_branch_added = True
                continue
            self.branches[key] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))
            self.num_branches += 1

        # token projections
        self.embed_dim = 64
        self.num_clinical = 37

        self.img_proj = nn.Linear(2048, self.embed_dim)
        self.clinical_proj = nn.Linear(1, self.embed_dim)

        """
        How it projects 1 scalar into 64 tokens is by taking the one clinical number and
        multiplying it by 64 different numbers, with each number representing a different aspect of 
        the clinical feature that the model learned is useful for cspca prediction.

        These numbers are randomly initialised when the model is created, and 
        progressively become helpful. Everytime the model makes a wrong prediction, backpropagation nudges 
        all 64 weights slightly so the next prediction is better. After thousands of patients, the weights converge to values that are genuinely useful.

        """

        # multi-head attention
        # total tokens = num_branches (2) + num_clinical (37) = 39
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=config["hyperparameters"]["dropout"],
            batch_first=True,
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        # classifier
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
        )

    def forward(self, data_dict, tabular_features):
        # imaging tokens
        img_tokens = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            feat = self.branches[branch_name](inputs)   # (batch, 2048)
            token = self.img_proj(feat)                 # (batch, 64)
            img_tokens.append(token.unsqueeze(1))       # (batch, 1, 64)

        img_tokens = torch.cat(img_tokens, dim=1)       # (batch, 2, 64)

        # clinical tokens
        clinical_tokens = []
        for i in range(self.num_clinical):
            feat = tabular_features[:, i].unsqueeze(1)  # (batch, 1)
            token = self.clinical_proj(feat)             # (batch, 64)
            clinical_tokens.append(token.unsqueeze(1))  # (batch, 1, 64)

        clinical_tokens = torch.cat(clinical_tokens, dim=1)  # (batch, 37, 64)

        # stack all tokens
        tokens = torch.cat([img_tokens, clinical_tokens], dim=1)  # (batch, 39, 64)

        # multi-head attention
        attended, _ = self.multihead_attn(tokens, tokens, tokens)
        attended = self.norm(attended + tokens)  # residual connection

        # pool and classify
        pooled = attended.mean(dim=1)   # (batch, 64)
        out = self.fc(pooled)           # (batch, 2)
        return out

    def training_step(self, batch, batch_idx):
        data_dict = batch["volume_data_dict"]
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(data_dict, tabular)
        loss = self.criterion(logits, target)
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch["volume_data_dict"]
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(data_dict, tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            if self.log_configs["log_run"]:
                self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def on_train_start(self):
        print("Freezing ResNet branches (permanent)...")
        for name, param in self.named_parameters():
            if "branches" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True