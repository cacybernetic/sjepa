"""Building blocks of the S-JEPA model.

Every component lives in its own file and has a single responsibility. This
package re-exports them so the rest of the code can import from one place.
"""

from .normalization import Fp32GroupNorm, Fp32LayerNorm
from .gradient_scaling import FeatureGradientScaler, scale_gradient
from .feature_extractor import ConvFeatureExtractor
from .positional_encoding import ConvPositionalEncoding
from .attention import MultiHeadSelfAttention
from .transformer import TransformerEncoderLayer
from .encoder import SpeechEncoder
from .predictor import JEPAPredictor
from .cluster_head import ClusterHead
from .masking import BlockMaskGenerator, build_padding_mask
from .ema import EmaEncoder, SwitchedEmaScheduler
from .losses import JEPAObjective, KLDivergenceLoss

__all__ = [
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "FeatureGradientScaler",
    "scale_gradient",
    "ConvFeatureExtractor",
    "ConvPositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerEncoderLayer",
    "SpeechEncoder",
    "JEPAPredictor",
    "ClusterHead",
    "BlockMaskGenerator",
    "build_padding_mask",
    "EmaEncoder",
    "SwitchedEmaScheduler",
    "JEPAObjective",
    "KLDivergenceLoss",
]
