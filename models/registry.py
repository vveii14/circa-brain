"""
Registry for pluggable components: image encoders, ROI encoders, fusion modules.
Add new backbones by registering and referring to them by name in config.
"""

import torch.nn as nn
from typing import Dict, Callable, Any, Optional


# ---------------------------------------------------------------------------
# Image encoders: name -> (constructor, default_kwargs)
IMAGE_ENCODERS: Dict[str, Callable[..., nn.Module]] = {}

# ROI encoders: name -> constructor (takes n_rois, feat_dim, ...)
ROI_ENCODERS: Dict[str, Callable[..., nn.Module]] = {}

# Fusion: name -> constructor (takes image_dim, roi_dim, num_classes, ...)
FUSION_MODULES: Dict[str, Callable[..., nn.Module]] = {}


def register_image_encoder(name: str):
    """Decorator: register an image encoder class. Must have feat_dim in __init__."""

    def _register(cls):
        IMAGE_ENCODERS[name] = cls
        return cls

    return _register


def register_roi_encoder(name: str):
    """Decorator: register an ROI encoder class. Must take n_rois, feat_dim."""

    def _register(cls):
        ROI_ENCODERS[name] = cls
        return cls

    return _register


def register_fusion(name: str):
    """Decorator: register a fusion module. Must take image_dim, roi_dim, num_classes."""

    def _register(cls):
        FUSION_MODULES[name] = cls
        return cls

    return _register


def build_image_encoder(name: str, feat_dim: int, **kwargs) -> nn.Module:
    if name not in IMAGE_ENCODERS:
        raise ValueError(f"Unknown image_encoder: {name}. Available: {list(IMAGE_ENCODERS.keys())}")
    return IMAGE_ENCODERS[name](feat_dim=feat_dim, **kwargs)


def build_roi_encoder(name: str, n_rois: int, feat_dim: int, **kwargs) -> nn.Module:
    if name not in ROI_ENCODERS:
        raise ValueError(f"Unknown roi_encoder: {name}. Available: {list(ROI_ENCODERS.keys())}")
    # Pass mode=name for encoders that use it (e.g. ROIEncoder)
    return ROI_ENCODERS[name](n_rois=n_rois, feat_dim=feat_dim, mode=name, **kwargs)


def build_fusion(name: str, image_dim: int, roi_dim: int, num_classes: int, **kwargs) -> nn.Module:
    if name not in FUSION_MODULES:
        raise ValueError(f"Unknown fusion: {name}. Available: {list(FUSION_MODULES.keys())}")
    return FUSION_MODULES[name](
        image_dim=image_dim,
        roi_dim=roi_dim,
        num_classes=num_classes,
        **kwargs,
    )
