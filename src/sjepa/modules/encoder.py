"""Speech encoder (the model `f_phi`).

The encoder maps a raw waveform to frame-level representations. It is the only
part kept after training; the predictor and cluster head are thrown away.

Pipeline:
  1. CNN frontend turns the waveform into frame features.
  2. A linear projection and layer norm bring them to the hidden size.
  3. Masked frames are zeroed (the predictor fills them later).
  4. A convolutional positional encoding adds order information.
  5. A stack of transformer layers builds the final representation.

The class focuses on one job: encode a waveform into frame representations.
"""

import torch
import torch.nn as nn

from .feature_extractor import ConvFeatureExtractor
from .gradient_scaling import FeatureGradientScaler
from .normalization import Fp32LayerNorm
from .positional_encoding import ConvPositionalEncoding
from .transformer import TransformerEncoderLayer


class SpeechEncoder(nn.Module):
    """HuBERT-style speech encoder with 6 transformer layers by default."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = ConvFeatureExtractor(config.conv_dim)
        self.feature_grad_scaler = FeatureGradientScaler(config.feature_grad_mult)
        self.post_extract_proj = nn.Linear(config.conv_dim, config.hidden_dim)
        self.layer_norm = Fp32LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_conv = ConvPositionalEncoding(config.hidden_dim,
                                               config.conv_pos_kernel,
                                               config.conv_pos_groups)
        self.encoder_layer_norm = Fp32LayerNorm(config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.hidden_dim, config.num_heads,
                                    config.ffn_dim, config.dropout,
                                    config.attention_dropout,
                                    config.activation_dropout)
            for _ in range(config.num_layers)
        ])
        self.layer_drop = config.layer_drop

    def _frontend(self, waveform):
        """Run the CNN frontend and bring features to the hidden size."""
        x = self.feature_extractor(waveform)
        x = self.feature_grad_scaler(x)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        return x

    @staticmethod
    def _trim(frame_mask, length):
        """Cut a frame mask to the real number of frames.

        The CNN output length can be one or two frames shorter than the
        analytic estimate. We trim the masks so every tensor lines up.
        """
        if frame_mask is None:
            return None
        return frame_mask[:, :length]

    @staticmethod
    def _apply_masks(x, mask, padding_mask):
        """Zero the masked frames and the padding frames.

        Args:
            x: features of shape (batch, length, hidden).
            mask: bool tensor, True where a frame is masked. May be None.
            padding_mask: bool tensor, True where a frame is real. May be None.

        Returns:
            The features with hidden and padded frames set to zero.
        """
        keep = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask is not None:
            keep = keep & (~mask)
        if padding_mask is not None:
            keep = keep & padding_mask
        return x * keep.unsqueeze(-1).to(x.dtype)

    def _should_drop_layer(self):
        """Decide if one layer is skipped (LayerDrop), only in training."""
        if not self.training or self.layer_drop <= 0.0:
            return False
        return torch.rand(1).item() < self.layer_drop

    def _run_transformer(self, x, padding_mask):
        """Run every transformer layer and collect each layer output."""
        outputs = []
        for layer in self.layers:
            if not self._should_drop_layer():
                x = layer(x, key_padding_mask=padding_mask)
            outputs.append(x)
        return outputs

    def forward(self, waveform, mask=None, padding_mask=None,
                return_layers=False):
        """Encode a waveform into frame representations.

        Args:
            waveform: tensor of shape (batch, 1, num_samples).
            mask: bool tensor (batch, num_frames), True where masked. May be None.
            padding_mask: bool tensor (batch, num_frames), True for real frames.
            return_layers: when True, return the list of all layer outputs.

        Returns:
            The last layer output of shape (batch, num_frames, hidden), or the
            list of all layer outputs when `return_layers` is True.
        """
        x = self._frontend(waveform)
        x = self.dropout(x)
        length = x.shape[1]
        mask = self._trim(mask, length)
        padding_mask = self._trim(padding_mask, length)
        x = self._apply_masks(x, mask, padding_mask)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        outputs = self._run_transformer(x, padding_mask)
        if return_layers:
            return outputs
        return outputs[-1]

    @torch.no_grad()
    def extract_layer(self, waveform, layer_index, padding_mask=None):
        """Return the clean features at one layer (no mask, no gradient).

        This is used by the EMA encoder to feed the online GMM and by the
        effective rank layer selection. The input is the clean waveform.
        """
        outputs = self.forward(waveform, mask=None, padding_mask=padding_mask,
                               return_layers=True)
        if not 0 <= layer_index < len(outputs):
            raise IndexError(
                f"layer_index {layer_index} is out of range "
                f"for {len(outputs)} layers")
        return outputs[layer_index]
