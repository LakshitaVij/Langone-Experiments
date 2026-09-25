import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch


class ClinicalChannelAttention(nn.Module):
    def __init__(self, channel_dim=2048, tabular_dim=37, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim + tabular_dim, channel_dim // reduction),
            nn.ReLU(),
            nn.Linear(channel_dim // reduction, channel_dim),
        )

    def forward(self, x, tabular_features):
        avg = x.mean(dim=[2, 3, 4])
        combined = torch.cat([avg, tabular_features], dim=1)
        weights = torch.sigmoid(self.mlp(combined))
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class ClinicalSpatialAttention(nn.Module):
    def __init__(self, tabular_dim=37):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.conv = nn.Conv3d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x, tabular_features):
        avg = x.mean(dim=1, keepdim=True)
        bias = self.mlp(tabular_features)
        bias = bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        spatial = avg + bias
        spatial = self.conv(spatial)
        weights = torch.sigmoid(spatial)
        return x * weights


class ResNetBranchEarly(ResNetBranch):
    def __init__(self, block, layers, in_chans, tabular_dim=37):
        super().__init__(block, layers, in_chans)
        self.channel_attn = ClinicalChannelAttention(
            channel_dim=256,  # layer1 output channels
            tabular_dim=tabular_dim
        )
        self.spatial_attn = ClinicalSpatialAttention(tabular_dim=tabular_dim)

    def forward(self, x, tabular_features):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)

        # CBAM after layer1 — finer spatial detail (13×45×45)
        out = self.channel_attn(out, tabular_features)
        out = self.spatial_attn(out, tabular_features)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class TriSeriesModel(Base3DResNet):
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
        self.feature_dim = 0
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    branch_name = "adc_b1500"
                    self.branches[branch_name] = ResNetBranchEarly(
                        Bottleneck, [3, 4, 6, 3], 2
                    )
                    self.branch_specs.append((branch_name, self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue

            branch_name = key
            self.branches[branch_name] = ResNetBranchEarly(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((branch_name, (key,)))
            self.feature_dim += 2048

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=False),
            self.dropout,
            nn.Linear(256, 2),
        )

        if "axt2" in self.branches:
            self.resnet_single_branch = self.branches["axt2"]
        elif self.branch_specs:
            self.resnet_single_branch = self.branches[self.branch_specs[0][0]]

        if self.stack_adc_b1500:
            if "adc_b1500" in self.branches:
                self.resnet_dual_branch1 = self.branches["adc_b1500"]
        else:
            if "adc" in self.branches:
                self.resnet_dual_branch1 = self.branches["adc"]
            if "b1500" in self.branches:
                self.resnet_dual_branch2 = self.branches["b1500"]

    def on_train_start(self):
        print("Freezing ResNet branches (permanent)...")
        for name, param in self.named_parameters():
            if "channel_attn" in name or "spatial_attn" in name or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, data_dict, tabular_features):
        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs, tabular_features))

        x = torch.cat(features, dim=1)
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