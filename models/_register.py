"""
Register the built-in encoders and fusion module for the
3D-SCTF (image) + NeuroGraph (ROI) contrastive model.
Import this module once (from models/__init__.py) to populate the registry.
"""

from . import registry
from .image_encoder_3dsctf import ImageEncoder3DSCTF
from .roi_encoder_brainnet import NeuroGraphROIEncoder
from .fusion import ContrastiveFusion


# Image encoder: 3D-SCTF (hybrid depthwise-separable conv + Transformer).
registry.IMAGE_ENCODERS["3dsctf"] = ImageEncoder3DSCTF

# ROI encoder: NeuroGraph / BrainNet residual GNN on the connectivity graph.
# "brainnet" is kept as an alias so checkpoints whose embedded cfg used that
# key load without modification (same class; the `mode` kwarg is unused).
registry.ROI_ENCODERS["neurograph"] = NeuroGraphROIEncoder
registry.ROI_ENCODERS["brainnet"] = NeuroGraphROIEncoder

# Fusion: cross-view contrastive alignment + concat classifier.
registry.FUSION_MODULES["contrastive"] = ContrastiveFusion
