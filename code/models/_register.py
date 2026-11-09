"""
Register built-in image encoders, ROI encoders, and fusion modules.
Import this module once (e.g. from models/__init__.py) to populate the registry.
"""

from . import registry
from .image_encoder import ImageEncoder3D, ImageEncoderResNet3D
from .image_encoder_vit3d import ImageEncoderViT3D
from .image_encoder_rae_vit import ImageEncoderRAEViT
from .image_encoder_3dsctf import ImageEncoder3DSCTF
from .roi_encoder import ROIEncoder
from .roi_encoder_brainnet import BrainNetROIEncoder
from .roi_encoder_chen2019 import Chen2019ROIEncoder
from .fusion import ConcatFusion, ContrastiveFusion, CrossAttentionFusion


# Image encoders: name -> class (build with feat_dim, in_channels=1, dropout=...)
registry.IMAGE_ENCODERS["conv3d"] = ImageEncoder3D
registry.IMAGE_ENCODERS["resnet3d"] = ImageEncoderResNet3D
registry.IMAGE_ENCODERS["vit3d"] = ImageEncoderViT3D
registry.IMAGE_ENCODERS["rae_vit_ad"] = ImageEncoderRAEViT
registry.IMAGE_ENCODERS["3dsctf"] = ImageEncoder3DSCTF

# ROI encoders: name = mode for ROIEncoder(n_rois, feat_dim, mode=name, ...)
for _mode in ("connectivity", "timeseries_1d", "transformer", "roi_vector"):
    registry.ROI_ENCODERS[_mode] = ROIEncoder
# BrainNet-style GNN encoder for ROI matrices / vectors
registry.ROI_ENCODERS["brainnet"] = BrainNetROIEncoder
# Chen et al. 2019-style DNN on single-atlas connectivity (flatten upper tri -> FC)
registry.ROI_ENCODERS["chen2019"] = Chen2019ROIEncoder

# Fusion: name -> class (build with image_dim, roi_dim, num_classes, ...)
registry.FUSION_MODULES["concat"] = ConcatFusion
registry.FUSION_MODULES["contrastive"] = ContrastiveFusion
registry.FUSION_MODULES["cross_attention"] = CrossAttentionFusion
