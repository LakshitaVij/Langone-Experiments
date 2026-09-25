import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch


class ClinicalEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.group_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
            nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
            nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, embed_dim)),
        ])

    def forward(self, tabular_features):
        group_indices = [
            (0, 5), (5, 9), (9, 12), (12, 20), (20, 28), (28, 36)
        ]
        group_embeddings = []
        for i, (start, end) in enumerate(group_indices):
            group = tabular_features[:, start:end]
            group_embeddings.append(self.group_mlps[i](group))
        return torch.cat(group_embeddings, dim=1)  # (batch, 384)


class FiLMGenerator(nn.Module):
    def __init__(self, clinical_dim=384, feature_dim=2048):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, clinical_emb):
        gamma = self.gamma_net(clinical_emb)  # (batch, 2048)
        beta = self.beta_net(clinical_emb)    # (batch, 2048)
        return gamma, beta


class FiLMModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 3

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)

        # imaging branches — full ResNet, frozen
        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self._dwi_branch_added = False

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self._dwi_branch_added = True
                continue
            self.branches[key] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))

        # clinical encoder — grouped MLPs → 384-dim
        self.clinical_encoder = ClinicalEncoder(embed_dim=64)

        # one FiLM generator per branch
        self.film_generators = nn.ModuleDict()
        for branch_name, _ in self.branch_specs:
            self.film_generators[branch_name] = FiLMGenerator(
                clinical_dim=384, feature_dim=2048
            )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # classifier: 4096 → 256 → 2
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(2048 * len(self.branch_specs), 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 2),
        )

        # backwards compatible attributes
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
            if "branches" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, data_dict, tabular_features):
        # clinical embedding
        clinical_emb = self.clinical_encoder(tabular_features)  # (batch, 384)

        # imaging features with FiLM modulation
        img_feats = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]

            # run through branch layers manually to intercept before avgpool
            branch = self.branches[branch_name]
            out = branch.conv1(inputs)
            out = branch.bn1(out)
            out = branch.relu(out)
            out = branch.maxpool(out)
            out = branch.layer1(out)
            out = branch.layer2(out)
            out = branch.layer3(out)
            out = branch.layer4(out)  # (batch, 2048, 7, 23, 23)

            # FiLM modulation: γ × F + β
            gamma, beta = self.film_generators[branch_name](clinical_emb)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch, 2048, 1, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = gamma * out + beta

            # avgpool after FiLM
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)  # (batch, 2048)
            img_feats.append(out)

        # concat and classify
        fused = torch.cat(img_feats, dim=1)  # (batch, 4096)
        out = self.fc(fused)
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