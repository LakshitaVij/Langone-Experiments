import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch


class ResNetBranchPatch(ResNetBranch):
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        return out  # (batch, 256, 13, 45, 45) — stop after layer1


class CrossAttentionFusionModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 3

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)

        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.num_branches = 0
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = ResNetBranchPatch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.num_branches += 1
                    self._dwi_branch_added = True
                continue
            self.branches[key] = ResNetBranchPatch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))
            self.num_branches += 1

        self.embed_dim = 64
        self.num_clinical = 6
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # patch CNN: 256 channels, 45×45 input → 64 channels, 3×3 output
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.embed_dim, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
        )

        self.clinical_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
            nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
            nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),
        ])

        # total tokens = 1 CLS + 234 img (117 per branch × 2) + 6 clinical = 241
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=config["hyperparameters"]["dropout"],
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.multihead_attn2 = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=config["hyperparameters"]["dropout"],
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
        )

    def forward(self, data_dict, tabular_features):
        # imaging tokens — vectorized across depth slices
        img_tokens = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            feat = self.branches[branch_name](inputs)  # (batch, 256, 13, H, W)

            # normalize spatial dims to 13×45×45
            feat = torch.nn.functional.adaptive_avg_pool3d(feat, (13, 45, 45))

            B, C, D, H, W = feat.shape
            feat_2d = feat.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)  # (batch*13, 256, 45, 45)
            patches = self.patch_cnn(feat_2d)           # (batch*13, 64, 3, 3)
            patches = patches.flatten(2)                 # (batch*13, 64, 9)
            patches = patches.permute(0, 2, 1)           # (batch*13, 9, 64)
            branch_tokens = patches.reshape(B, D*9, 64)  # (batch, 117, 64)
            img_tokens.append(branch_tokens)

        img_tokens = torch.cat(img_tokens, dim=1)  # (batch, 234, 64)

        # clinical tokens
        group_indices = [
            (0, 5),
            (5, 9),
            (9, 12),
            (12, 20),
            (20, 28),
            (28, 36),
        ]

        clinical_tokens = []
        for i, (start, end) in enumerate(group_indices):
            group = tabular_features[:, start:end]
            token = self.clinical_projs[i](group)
            clinical_tokens.append(token.unsqueeze(1))

        clinical_tokens = torch.cat(clinical_tokens, dim=1)  # (batch, 6, 64)

        # prepend CLS token
        batch_size = img_tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, 64)

        tokens = torch.cat([cls, img_tokens, clinical_tokens], dim=1)  # (batch, 241, 64)

        # cascade attention block 1
        attended, _ = self.multihead_attn(tokens, tokens, tokens)
        tokens = self.norm(attended + tokens)

        # cascade attention block 2
        attended, _ = self.multihead_attn2(tokens, tokens, tokens)
        tokens = self.norm2(attended + tokens)

        # classify from CLS token
        cls_out = tokens[:, 0, :]
        out = self.fc(cls_out)
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