"""
Cross-attention fusion model for csPCa detection.
Tabular features attend over imaging branch features via multi-head attention.

Architecture:
- Two ResNet branches (T2 + stacked ADC/b1500) → image features (batch, num_branches, 2048)
- Tabular MLP (17 → 64 → 32) → tabular features (batch, 32)
- Cross-attention: tabular queries, image keys/values → attended image (batch, 64)
- Classifier: attended image + tabular → 2 classes
"""
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

        # ── imaging branches (identical to TriSeriesModel) ───────────────────
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

        # ── tabular encoder ──────────────────────────────────────────────────
        self.tabular_encoder = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ── cross-attention ──────────────────────────────────────────────────
        # query: tabular (32-dim), key/value: image branches (2048-dim each)
        # project all to same dim (64) for attention
        self.query_proj = nn.Linear(32, 64)
        self.key_proj = nn.Linear(2048, 64)
        self.value_proj = nn.Linear(2048, 64)
        self.attn_scale = 64 ** 0.5

        # ── classifier ───────────────────────────────────────────────────────
        # attended image (64) + tabular (32) → 2 classes
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
        )

        # backwards-compatible attributes for existing utilities (e.g. Grad-CAM)
        if "axt2" in self.branches:
            self.resnet_single_branch = self.branches["axt2"]
        elif self.branch_specs:
            self.resnet_single_branch = self.branches[self.branch_specs[0][0]]
        if self.stack_adc_b1500 and "adc_b1500" in self.branches:
            self.resnet_dual_branch1 = self.branches["adc_b1500"]

    def forward(self, data_dict, tabular_features):
        # ── 1. extract image features from each branch ───────────────────────
        branch_features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            branch_features.append(self.branches[branch_name](inputs))
        # stack into (batch, num_branches, 2048)
        img = torch.stack(branch_features, dim=1)

        # ── 2. encode tabular features ───────────────────────────────────────
        tab = self.tabular_encoder(tabular_features)  # (batch, 32)

        # ── 3. cross-attention ───────────────────────────────────────────────
        # query: tabular → (batch, 1, 64)
        Q = self.query_proj(tab).unsqueeze(1)
        # keys/values: image branches → (batch, num_branches, 64)
        K = self.key_proj(img)
        V = self.value_proj(img)

        # attention scores: (batch, 1, num_branches)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # attended image: (batch, 1, 64) → (batch, 64)
        attended = torch.bmm(attn_weights, V).squeeze(1)

        # ── 4. classify ──────────────────────────────────────────────────────
        x = torch.cat([attended, tab], dim=1)  # (batch, 96)
        out = self.fc(x)
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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
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
            print("Test Loss", loss)
        return {"test_loss": loss}

    def on_train_start(self):
        print("Freezing ResNet branches for first 10 epochs...")
        for name, param in self.named_parameters():
            if not name.startswith("fc") and \
               not name.startswith("tabular_encoder") and \
               not name.startswith("query_proj") and \
               not name.startswith("key_proj") and \
               not name.startswith("value_proj"):
                param.requires_grad = False

    def on_train_epoch_start(self):
        if self.current_epoch == 10:
            print("Unfreezing all parameters...")
            for param in self.parameters():
                param.requires_grad = True