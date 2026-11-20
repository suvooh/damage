"""DINOv2 damage segmentation model and checkpoint loader.

This project contains only the selected vehicle-damage model: a DINOv2
ViT-S/14 backbone with decoupled damage-presence, damage-type, and
vehicle-exterior dense heads. Other architectures that existed in the source
repository have been removed.
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F


APPROVED_DINOV2_BACKBONES = {"vit_small_patch14_dinov2.lvd142m"}
SUPPORTED_ARCHITECTURE = "dinov2"


class Dinov2DamageSegmenter(nn.Module):
    """DINOv2 patch features with lightweight dense prediction heads."""

    def __init__(
        self,
        num_classes: int,
        *,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
    ) -> None:
        super().__init__()
        if backbone_name not in APPROVED_DINOV2_BACKBONES:
            raise ValueError("DINOv2 backbone is not in the commercially reviewed allowlist")
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one damage type")
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        feature_dim = int(self.backbone.num_features)
        patch_size = self.backbone.patch_embed.patch_size
        self.patch_size = (
            (int(patch_size), int(patch_size))
            if isinstance(patch_size, int)
            else (int(patch_size[0]), int(patch_size[1]))
        )
        hidden = min(256, feature_dim)

        def head(channels: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(feature_dim, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden), hidden),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )

        self.presence_head = head(1)
        self.type_head = head(num_classes - 1)
        self.exterior_head = head(1)
        self.num_classes = num_classes

    def _extract_patch_features(self, padded: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(padded)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, prefix_tokens:]
        patch_height, patch_width = self.patch_size
        grid_height = padded.shape[-2] // patch_height
        grid_width = padded.shape[-1] // patch_width
        if patch_tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError("DINOv2 patch-token shape does not match the input grid")
        return patch_tokens.transpose(1, 2).reshape(
            padded.shape[0], patch_tokens.shape[2], grid_height, grid_width
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        original_height, original_width = images.shape[-2:]
        patch_height, patch_width = self.patch_size
        pad_height = (-original_height) % patch_height
        pad_width = (-original_width) % patch_width
        padded = F.pad(images, (0, pad_width, 0, pad_height))
        features = self._extract_patch_features(padded)

        def upsample(head: nn.Module, value: torch.Tensor) -> torch.Tensor:
            value = F.interpolate(
                head(value),
                size=padded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return value[..., :original_height, :original_width]

        return {
            "presence": upsample(self.presence_head, features),
            "type": upsample(self.type_head, features),
            "exterior": upsample(self.exterior_head, features),
        }


def damage_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compose mutually exclusive semantic probabilities from decoupled heads."""
    presence = outputs["presence"].sigmoid()
    damage_type = outputs["type"].softmax(dim=1)
    return torch.cat((1.0 - presence, presence * damage_type), dim=1)


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load the pinned damage checkpoint; other model architectures are rejected."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint.get(
        "architecture",
        checkpoint.get("config", {}).get("architecture", SUPPORTED_ARCHITECTURE),
    )
    if architecture != SUPPORTED_ARCHITECTURE:
        raise ValueError(
            f"unsupported checkpoint architecture {architecture!r}; "
            f"this project only serves {SUPPORTED_ARCHITECTURE!r}"
        )
    if "classes" not in checkpoint or "model" not in checkpoint:
        raise ValueError("checkpoint is missing its classes or model state")
    model = Dinov2DamageSegmenter(
        len(checkpoint["classes"]),
        backbone_name=checkpoint.get("config", {}).get(
            "backbone_name", "vit_small_patch14_dinov2.lvd142m"
        ),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint
