import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TwoHeadNet(nn.Module):
    def __init__(self, n_crop: int, n_health: int):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_ch = 576
        self.crop_head = nn.Linear(in_ch, n_crop)
        self.health_head = nn.Linear(in_ch, n_health)
        self.gate_head = nn.Linear(in_ch, 1)

    def embed(self, x):
        z = self.features(x)
        return self.avgpool(z).flatten(1)

    def forward(self, x):
        z = self.embed(x)
        return self.crop_head(z), self.health_head(z), self.gate_head(z)

    def freeze_backbone(self, freeze: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = not freeze

    def unfreeze_last(self, n_blocks: int = 4) -> None:
        self.freeze_backbone(True)
        blocks = list(self.features.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True


def load_weights(model: TwoHeadNet, ckpt: dict) -> None:
    model.load_state_dict(ckpt["model"], strict=False)
