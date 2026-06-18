# Populate registry with the built-in encoders and fusion (must run before build_*)
from . import _register  # noqa: F401

from . import registry
from .fusion_model import ImageROIFusionModel
from .image_encoder_3dsctf import ImageEncoder3DSCTF
from .roi_encoder_brainnet import NeuroGraphROIEncoder
from .fusion import ContrastiveFusion

__all__ = [
    "registry",
    "ImageROIFusionModel",
    "ImageEncoder3DSCTF",
    "NeuroGraphROIEncoder",
    "ContrastiveFusion",
]
