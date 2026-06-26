"""Train S-JEPA from a YAML config.

Usage:
    trainsjepa -c cpu/configs/train.yaml
    trainsjepa --config cpu/configs/train.yaml

The program runs three stages: training, validation on a fraction of the test
set after each epoch, and a final evaluation on the whole test set. It saves
checkpoints, the best and last weights, the history CSV, and the history plots.
"""

from __future__ import annotations

from ..assembly import PipelineBuilder
from ..logging import get_logger
from .common import parse_config_arg, setup_run

_LOGGER = get_logger()


def run(config_path):
    """Build the trainer from a config path and run the full pipeline."""
    config, layout, resumed = setup_run(config_path, "train")
    _LOGGER.info("Run folder reused for resume: {}", resumed)
    trainer = PipelineBuilder(config, layout).build()
    trainer.run()
    trainer.final_evaluate()
    _LOGGER.info("All done. Outputs are in {}", layout.root)


def main():
    """Console entry point for the trainsjepa command."""
    config_path = parse_config_arg("Train the S-JEPA speech model")
    run(config_path)


if __name__ == "__main__":
    main()
