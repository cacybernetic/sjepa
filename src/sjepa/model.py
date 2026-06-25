"""The complete S-JEPA model.

This file puts the three trained parts together:
  * the encoder `f_phi` (kept for downstream use),
  * the predictor `h_psi` (fills the masked frames, training only),
  * the cluster head `g_omega` (maps frames to K logits, training only).

The model returns cluster logits at visible frames (from the encoder) and at
masked frames (from the predictor). Only the masked-frame logits drive the
loss; the visible ones are kept for analysis. The EMA encoder and the GMM live
outside this file because they belong to the training loop, not the network.

The model is built for stable and efficient training: float32 normalization,
fused attention, gradient scaling on the CNN frontend, and a careful weight
initialization.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import SJEPAConfig
from .logging import get_logger, log_hparams
from .modules.encoder import SpeechEncoder
from .modules.predictor import JEPAPredictor
from .modules.cluster_head import ClusterHead
from .modules.ema import EmaEncoder, SwitchedEmaScheduler

_LOGGER = get_logger()


# Ready-made model sizes. Each entry holds the settings that change between
# sizes. The "base" size is the one used in the paper (6 layers, 51.8M encoder
# parameters). The "large" size matches HuBERT-Base depth (12 layers), which is
# the default of the reference code. The smaller sizes help quick experiments
# and tests on a single machine.
_SIZE_PRESETS = {
    "tiny": dict(
        conv_dim=128, hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=256,
        predictor_heads=4, predictor_ffn_dim=256, max_frames=256,
    ),
    "small": dict(
        conv_dim=512, hidden_dim=512, num_layers=4, num_heads=8, ffn_dim=2048,
        predictor_heads=8, predictor_ffn_dim=1024,
    ),
    "base": dict(
        conv_dim=512, hidden_dim=768, num_layers=6, num_heads=12, ffn_dim=3072,
        predictor_heads=8, predictor_ffn_dim=1536,
    ),
    "large": dict(
        conv_dim=512, hidden_dim=768, num_layers=12, num_heads=12, ffn_dim=3072,
        predictor_heads=8, predictor_ffn_dim=1536,
    ),
}

# The list of valid size names, handy for error messages and tests.
SJEPA_SIZES = tuple(_SIZE_PRESETS.keys())


def build_config(size="base", **overrides):
    """Build a model config for a named size.

    Args:
        size: one of the names in `SJEPA_SIZES` ("tiny", "small", "base",
            "large").
        overrides: extra settings that replace the preset values. For example
            `num_clusters=500` for phase 2.

    Returns:
        A validated `SJEPAConfig`.
    """
    if size not in _SIZE_PRESETS:
        raise ValueError(
            f"unknown size '{size}', choose one of {SJEPA_SIZES}")
    params = dict(_SIZE_PRESETS[size])
    params.update(overrides)
    _LOGGER.info("Building config for size '{}' with {} overrides",
                 size, len(overrides))
    return SJEPAConfig(**params)


def build_model(size="base", **overrides):
    """Build a complete S-JEPA model for a named size.

    This is the simple entry point for users who do not want to make a config
    by hand. It is the same as `SJEPA.from_size`.

    Args:
        size: one of the names in `SJEPA_SIZES`.
        overrides: extra settings that replace the preset values.

    Returns:
        A ready `SJEPA` model.
    """
    return SJEPA.from_size(size, **overrides)


@dataclass
class SJEPAOutput:
    """The result of one forward pass.

    Fields:
        logits_visible: cluster logits from the encoder output.
        logits_masked: cluster logits from the predictor output.
        encoder_output: the encoder frame representations.
        predictor_output: the predictor frame representations.
        mask: the masked-frame flags aligned to the encoder length.
        padding_mask: the real-frame flags aligned to the encoder length.
    """

    logits_visible: torch.Tensor
    logits_masked: torch.Tensor
    encoder_output: torch.Tensor
    predictor_output: torch.Tensor
    mask: torch.Tensor
    padding_mask: torch.Tensor


class SJEPA(nn.Module):
    """The full S-JEPA network (encoder, predictor, cluster head)."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SJEPAConfig()
        self.encoder = SpeechEncoder(self.config)
        self.predictor = JEPAPredictor(self.config)
        self.cluster_head = ClusterHead(self.config.hidden_dim,
                                        self.config.num_clusters)
        self.apply(self._init_weights)
        self._log_summary()

    @classmethod
    def from_size(cls, size="base", **overrides):
        """Build a model from a named size preset.

        Args:
            size: one of the names in `SJEPA_SIZES` ("tiny", "small", "base",
                "large").
            overrides: extra settings that replace the preset values, for
                example `num_clusters=500` for phase 2.

        Returns:
            A ready `SJEPA` model.
        """
        config = build_config(size, **overrides)
        return cls(config)

    @staticmethod
    def _init_weights(module):
        """Initialize weights for stable training.

        Linear and embedding weights use a small normal spread. Biases start at
        zero. Normalization layers keep their default init. This follows the
        common transformer recipe and avoids large early activations.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, waveform, mask, padding_mask=None):
        """Run the full training forward pass.

        Args:
            waveform: tensor of shape (batch, 1, num_samples).
            mask: bool tensor (batch, num_frames), True where masked.
            padding_mask: bool tensor (batch, num_frames), True for real frames.

        Returns:
            An `SJEPAOutput` with the encoder and predictor logits.
        """
        encoder_output = self.encoder(waveform, mask=mask,
                                      padding_mask=padding_mask)
        # The CNN length can be a little shorter than the analytic estimate,
        # so we align the masks to the real encoder length before going on.
        length = encoder_output.shape[1]
        mask = mask[:, :length]
        if padding_mask is not None:
            padding_mask = padding_mask[:, :length]
        predictor_output = self.predictor(encoder_output, mask,
                                          padding_mask=padding_mask)
        logits_visible = self.cluster_head(encoder_output)
        logits_masked = self.cluster_head(predictor_output)
        return SJEPAOutput(
            logits_visible=logits_visible,
            logits_masked=logits_masked,
            encoder_output=encoder_output,
            predictor_output=predictor_output,
            mask=mask,
            padding_mask=padding_mask,
        )

    @torch.no_grad()
    def extract_features(self, waveform, layer_index=-1, padding_mask=None):
        """Return clean encoder features for inference or probing.

        The encoder is used without any mask. This is the path used after
        training, when only `f_phi` is kept.

        Args:
            waveform: tensor of shape (batch, 1, num_samples).
            layer_index: which transformer layer to read (-1 is the last one).
            padding_mask: bool tensor (batch, num_frames), True for real frames.

        Returns:
            A tensor of shape (batch, num_frames, hidden_dim).
        """
        layers = self.encoder(waveform, mask=None, padding_mask=padding_mask,
                              return_layers=True)
        return layers[layer_index]

    def build_ema_encoder(self, scheduler=None):
        """Create the EMA encoder used in phase 2.

        Args:
            scheduler: an optional `SwitchedEmaScheduler`. A default one is
                built when none is given.

        Returns:
            An `EmaEncoder` that wraps a frozen copy of the current encoder.
        """
        scheduler = scheduler or SwitchedEmaScheduler()
        return EmaEncoder(self.encoder, scheduler=scheduler)

    def set_num_clusters(self, num_clusters):
        """Rebuild the cluster head for a new number of clusters.

        Phase 2 changes K from 100 to 500, so the cluster head must be made
        again. The encoder and predictor are not touched.
        """
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0")
        self.config.num_clusters = num_clusters
        self.cluster_head = ClusterHead(self.config.hidden_dim, num_clusters)
        self.cluster_head.apply(self._init_weights)
        _LOGGER.info("Cluster head rebuilt with K={}", num_clusters)

    def count_parameters(self):
        """Count parameters of each part and of the whole model.

        Returns:
            A dict that maps a part name to its parameter count in millions.
        """
        def millions(module):
            return sum(p.numel() for p in module.parameters()) / 1e6

        return {
            "encoder_M": millions(self.encoder),
            "predictor_M": millions(self.predictor),
            "cluster_head_M": millions(self.cluster_head),
            "total_M": millions(self),
        }

    def _log_summary(self):
        """Print the model configuration and parameter counts."""
        self.config.summary()
        counts = self.count_parameters()
        rounded = {key: round(value, 2) for key, value in counts.items()}
        log_hparams("SJEPA parameter counts (millions)", rounded, color="green")
