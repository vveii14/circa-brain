# Populate registry with built-in encoders and fusion (must run before build_*)
from . import _register  # noqa: F401

from . import registry
from .fusion_model import ImageROIFusionModel
from .image_encoder import ImageEncoder3D, ImageEncoderResNet3D
from .roi_encoder import ROIEncoder
from .fusion import ConcatFusion, ContrastiveFusion, CrossAttentionFusion

__all__ = [
    "registry",
    "ImageROIFusionModel",
    "ImageEncoder3D",
    "ImageEncoderResNet3D",
    "ROIEncoder",
    "ConcatFusion",
    "ContrastiveFusion",
    "CrossAttentionFusion",
]
