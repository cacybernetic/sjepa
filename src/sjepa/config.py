"""Configuration for the S-JEPA model.

This module holds one dataclass, `SJEPAConfig`, that groups every setting of
the model in one place. Keeping all settings together makes the code easy to
read and easy to change. The default values follow the paper (6 transformer
layers, 768 hidden dimension, 51.8M parameters target).

The config also checks its own values so a wrong setting is caught early,
before any heavy computation starts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .logging import log_hparams


@dataclass
class SJEPAConfig:
    """All hyperparameters of the S-JEPA network.

    The fields are grouped by component: CNN frontend, transformer encoder,
    predictor, and cluster head. Phase 1 uses `num_clusters = 100` and phase 2
    uses `num_clusters = 500`.
    """

    # --- CNN feature extractor (waveform to frames) ---
    conv_dim: int = 512
    feature_grad_mult: float = 0.1

    # --- Transformer encoder (the part kept for downstream use) ---
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    layer_drop: float = 0.05
    conv_pos_kernel: int = 128
    conv_pos_groups: int = 16

    # --- Predictor (fills the masked frames during training only) ---
    predictor_layers: int = 1
    predictor_heads: int = 8
    predictor_ffn_dim: int = 1536
    predictor_dropout: float = 0.1
    max_frames: int = 750

    # --- Cluster head (maps frames to K logits) ---
    num_clusters: int = 100

    # --- Frame rate ---
    hop: int = 320

    def __post_init__(self) -> None:
        """Check the values right after the object is built."""
        self.validate()

    def validate(self) -> None:
        """Make sure the settings are valid and consistent.

        We raise a clear error when a value is wrong. This stops the program
        early instead of failing deep inside a tensor operation.
        """
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must divide by num_heads: "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"
            )
        if self.hidden_dim % self.predictor_heads != 0:
            raise ValueError(
                "hidden_dim must divide by predictor_heads: "
                f"got hidden_dim={self.hidden_dim}, "
                f"predictor_heads={self.predictor_heads}"
            )
        positive_fields = {
            "conv_dim": self.conv_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "predictor_layers": self.predictor_layers,
            "num_clusters": self.num_clusters,
            "max_frames": self.max_frames,
            "hop": self.hop,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, but got {value}")

    @property
    def head_dim(self) -> int:
        """Return the size of one attention head in the encoder."""
        return self.hidden_dim // self.num_heads

    def summary(self) -> None:
        """Print all settings to the terminal for traceability."""
        log_hparams("SJEPA configuration", asdict(self))
