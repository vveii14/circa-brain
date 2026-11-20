"""Register the encoders and fusion heads by name (imported once from models/__init__.py)."""

from . import registry
from .fusion import ConcatFusion, ContrastiveFusion
from .image_encoder_3dsctf import ImageEncoder3DSCTF
from .roi_encoder_neurograph import NeuroGraphROIEncoder

# Imaging branch: 3DSC-TF (3D depthwise-separable convolution + Transformer).
registry.IMAGE_ENCODERS["3dsctf"] = ImageEncoder3DSCTF

# ROI branch: NeuroGraph residual GNN on the ROI connectivity graph.
registry.ROI_ENCODERS["neurograph"] = NeuroGraphROIEncoder

# Fusion heads.
registry.FUSION_MODULES["contrastive"] = ContrastiveFusion  # proposed: cross-view contrastive alignment
registry.FUSION_MODULES["concat"] = ConcatFusion            # ablation: plain concatenation
