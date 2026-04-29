"""Model factory: CustomCNN, MobileNetV2, and ConvNeXt-Tiny."""
import torch.nn as nn
import torchvision.models as tv_models


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class CustomCNN(nn.Module):
    """4-block CNN baseline (~1 M params)."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        self.features = nn.Sequential(
            *[_ConvBlock(channels[i], channels[i + 1]) for i in range(4)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        x = x.flatten(1)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Return a model ready for training.

    Args:
        name: One of 'custom_cnn', 'mobilenet_v2', 'convnext_tiny'.
        num_classes: Output classes.
        pretrained: Load ImageNet weights for transfer learning models.

    Returns:
        Initialised nn.Module.
    """
    if name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes)
        print(f"[MODEL] CustomCNN — {sum(p.numel() for p in model.parameters()):,} params")
        return model

    if name == "mobilenet_v2":
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        n = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] MobileNetV2 — {n:,} params (pretrained={pretrained})")
        return model

    if name == "convnext_tiny":
        import timm
        model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
        n = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] ConvNeXt-Tiny — {n:,} params (pretrained={pretrained})")
        return model

    raise ValueError(f"Unknown model '{name}'. Choose from: custom_cnn, mobilenet_v2, convnext_tiny")


def freeze_backbone(model: nn.Module, name: str) -> None:
    """Freeze all layers except the classification head."""
    if name == "mobilenet_v2":
        for param in model.features.parameters():
            param.requires_grad = False
    elif name == "convnext_tiny":
        for param_name, param in model.named_parameters():
            if "head" not in param_name:
                param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter."""
    for param in model.parameters():
        param.requires_grad = True
