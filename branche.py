"""
This module defines PyTorch Lightning ResNet3D implementations that accept multiple 
volume inputs.

All classes in this module inherit from the Base3DResNet class, and consist of 
multi-branch architectures. Each branch accepts a single volume input, and the 
outputs of each branch are concatenated along the feature dimension and passed 
through a fully connected layer.

Amritha Musipatla
Dec 25 2023
"""
import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch


class DualSeriesModel(Base3DResNet):
    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 2

        self.feature_dim = 2048

        if {'adc', 'b1500'} == set(self.series):
            if self.stack_adc_b1500:
                self.resnet_dual_branch1 = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
            else:
                self.resnet_dual_branch1 = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
                self.resnet_dual_branch2 = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
                self.feature_dim = self.feature_dim * 2
        else:
            self.resnet_dual_branch1 = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.resnet_dual_branch2 = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.feature_dim = self.feature_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 2),
        )

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])

    def forward(self, data_dict):
        if {"adc", "b1500"} == set(self.series):
            if self.stack_adc_b1500:
                adc_b1500 = torch.cat(
                    [data_dict["adc"], data_dict["b1500"]],
                    dim=1
                )
                x = self.resnet_dual_branch1(adc_b1500)
            else:
                x1 = self.resnet_dual_branch1(data_dict["adc"])
                x2 = self.resnet_dual_branch2(data_dict["b1500"])
                x = torch.cat((x1, x2), dim=1)
        else:
            volumes = [data_dict[key] for key in self.series]

            assert len(volumes) == 2

            x1 = self.resnet_dual_branch1(volumes[0])
            x2 = self.resnet_dual_branch2(volumes[1])
            x = torch.cat((x1, x2), dim=1)

        out = self.fc(x)
        return out


class TriSeriesModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 3

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(
            series_set
        )

        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    branch_name = "adc_b1500"
                    self.branches[branch_name] = ResNetBranch(
                        Bottleneck, [3, 4, 6, 3], 2
                    )
                    self.branch_specs.append((branch_name, self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue

            branch_name = key
            self.branches[branch_name] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((branch_name, (key,)))
            self.feature_dim += 2048

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=False),
            self.dropout,
            nn.Linear(256, 2),
        )

        # Backwards-compatible attributes for existing utilities (e.g., Grad-CAM)
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

    def forward(self, data_dict):
        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs))

        x = torch.cat(features, dim=1)
        out = self.fc(x)
        return out


class QuadSeriesModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 4

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(
            series_set
        )

        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    branch_name = "adc_b1500"
                    self.branches[branch_name] = ResNetBranch(
                        Bottleneck, [3, 4, 6, 3], 2
                    )
                    self.branch_specs.append((branch_name, self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue

            branch_name = key
            self.branches[branch_name] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
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

    def forward(self, data_dict):
        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs))

        x = torch.cat(features, dim=1)
        out = self.fc(x)
        return out