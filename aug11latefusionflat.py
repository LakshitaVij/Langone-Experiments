import torch
import torch.nn as nn
from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch

class FrozenClinicalEncoder(nn.Module):
    def __init__(self, clinical_ckpt=None):
        super().__init__()
        self.fc1 = nn.Linear(37, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        if clinical_ckpt is not None:
            state = torch.load(clinical_ckpt, map_location='cpu')
            self.fc1.weight.data = state['fc1.weight']
            self.fc1.bias.data = state['fc1.bias']
            self.fc2.weight.data = state['fc2.weight']
            self.fc2.bias.data = state['fc2.bias']
            print(f"[FrozenClinicalEncoder] Loaded from {clinical_ckpt}")
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x

class LateFusionFlatFrozen(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")
    
    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        
        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)
        
        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        self._dwi_branch_added = False
        
        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue
            self.branches[key] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))
            self.feature_dim += 2048
        
        self.img_proj = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.ReLU())
        
        clinical_ckpt = config["model_weights"].get("clinical_ckpt", None)
        self.clinical_encoder = FrozenClinicalEncoder(clinical_ckpt=clinical_ckpt)
        
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.ReLU(), self.dropout, nn.Linear(128, 2)
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
        print("Freezing ResNet and clinical encoder...")
        for name, param in self.named_parameters():
            if "branches" in name or "clinical_encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.branches.eval()
    
    def forward(self, data_dict, tabular_features):
        img_feats = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            img_feats.append(self.branches[branch_name](inputs))
        
        img_vec = torch.cat(img_feats, dim=1)
        img_emb = self.img_proj(img_vec)
        clin_emb = self.clinical_encoder(tabular_features)
        if not hasattr(self, "_debug_printed"):
            print(f"DEBUG clin_emb sample: {clin_emb[0][:5]}")
            print(f"DEBUG clin_emb std across batch: {clin_emb.std(dim=0).mean().item():.6f}")
            print(f"DEBUG img_emb std across batch: {img_emb.std(dim=0).mean().item():.6f}")
            self._debug_printed = True
        
        fused = torch.cat([img_emb, clin_emb], dim=1)
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
