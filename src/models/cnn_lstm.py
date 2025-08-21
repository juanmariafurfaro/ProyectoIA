from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm


class CNNEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, train_backbone: bool = False):
        super().__init__()
        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "googlenet":
            net = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT if pretrained else None)
            feat_dim = 1024
        else:
            raise ValueError("Backbone no soportado")

        # Quitar la clasificación final
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])  # hasta avgpool
        self.out_dim = feat_dim
        self.train_backbone = train_backbone
        if not train_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]
        feats = self.feature_extractor(x)  # [B*T, feat_dim, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B*T, feat_dim]
        return feats


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = "resnet18", hidden_size: int = 256,
                 num_layers: int = 1, bidirectional: bool = False, pretrained_backbone: bool = True,
                 train_backbone: bool = False, dropout: float = 0.3):
        super().__init__()
        self.encoder = CNNEncoder(backbone, pretrained_backbone, train_backbone)
        self.hidden_size = hidden_size
        self.num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.encoder.out_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_dirs, num_classes)
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [B, T, C, H, W]
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)
        feats = self.encoder(x)               # [B*T, F]
        feats = feats.view(B, T, -1)          # [B, T, F]
        out, (hn, cn) = self.lstm(feats)      # hn: [num_layers*num_dirs, B, H]
        last = out[:, -1, :]                  # usar el último *time step*
        logits = self.classifier(last)        # [B, num_classes]
        return logits