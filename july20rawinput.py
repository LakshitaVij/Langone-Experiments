import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet


class PatchEmbedder(nn.Module):
    def __init__(self, in_chans, embed_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        out = self.cnn(x)
        return out.view(out.size(0), -1)  # (batch, embed_dim)


class RawPatchFusionModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        self.series = [series.value["key"] for series in config["data"]["series"]]
        assert len(self.series) == 3

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)

        # patch embedders — one per branch
        # T2: 1 channel, ADC/b1500: 2 channels stacked
        self.embed_dim = 64
        self.patch_size = (5, 36, 36)  # D×H×W per patch
        self.grid = (6, 5, 5)          # 6×5×5 = 150 patches per branch

        self.t2_embedder = PatchEmbedder(in_chans=1, embed_dim=self.embed_dim)
        self.dwi_embedder = PatchEmbedder(in_chans=2, embed_dim=self.embed_dim)

        # clinical token projections — 6 groups
        self.num_clinical = 6
        self.clinical_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 1: PSA/volume
            nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 2: anatomy/history
            nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 3: global lesion
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 4: lesion 1
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 5: lesion 2
            nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.embed_dim)),  # Group 6: lesion 3
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # total tokens = 1 CLS + 300 img (150 per branch × 2) + 6 clinical = 307
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

        # classifier
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
        )

    def _extract_patches(self, vol, embedder):
        """Divide volume into patches and embed each one."""
        B, C, D, H, W = vol.shape
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid

        tokens = []
        for d in range(gd):
            for h in range(gh):
                for w in range(gw):
                    patch = vol[
                        :, :,
                        d*pd:(d+1)*pd,
                        h*ph:(h+1)*ph,
                        w*pw:(w+1)*pw,
                    ]  # (batch, C, 5, 36, 36)
                    token = embedder(patch)          # (batch, 64)
                    tokens.append(token.unsqueeze(1))  # (batch, 1, 64)

        return torch.cat(tokens, dim=1)  # (batch, 150, 64)

    def forward(self, data_dict, tabular_features):
        # imaging tokens
        t2 = data_dict["axt2"]                                      # (batch, 1, 30, 180, 180)
        dwi = torch.cat([data_dict["adc"], data_dict["b1500"]], dim=1)  # (batch, 2, 30, 180, 180)

        t2_tokens = self._extract_patches(t2, self.t2_embedder)      # (batch, 150, 64)
        dwi_tokens = self._extract_patches(dwi, self.dwi_embedder)   # (batch, 150, 64)

        img_tokens = torch.cat([t2_tokens, dwi_tokens], dim=1)       # (batch, 300, 64)

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
        cls = self.cls_token.expand(batch_size, -1, -1)      # (batch, 1, 64)

        tokens = torch.cat([cls, img_tokens, clinical_tokens], dim=1)  # (batch, 307, 64)

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
        print("Training from scratch — no frozen branches.")
        for name, param in self.named_parameters():
            param.requires_grad = True