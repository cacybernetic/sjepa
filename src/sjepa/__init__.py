"""S-JEPA: speech self-supervised model (reimplementation).

This package holds the model and its building blocks. Python is used for
training; inference is planned in Rust with numerical parity tests.

Public entry points:
  * `SJEPAConfig`: all model hyperparameters in one dataclass.
  * `SJEPA`: the complete model (encoder, predictor, cluster head).
  * `SJEPAOutput`: the dataclass returned by a forward pass.
  * `build_model` / `build_config`: builders for named model sizes.
  * `SJEPA_SIZES`: the list of valid size names.
"""

from .config import SJEPAConfig
from .model import SJEPA, SJEPAOutput, build_model, build_config, SJEPA_SIZES

__all__ = [
    "SJEPAConfig",
    "SJEPA",
    "SJEPAOutput",
    "build_model",
    "build_config",
    "SJEPA_SIZES",
]

__version__ = "0.1.0"
