"""Evaluate a trained S-JEPA model on the whole test set.

Usage:
    evalsjepa -c cpu/configs/eval.yaml

The config points `init_weights` at the weight file to evaluate (for example a
"best.pt"). The program builds the model and the targets, runs the metrics over
the full test set, and writes the values to results.csv in the eval folder.
"""

from __future__ import annotations

import csv
import os

from ..assembly import PipelineBuilder
from ..logging import get_logger
from .common import parse_config_arg, setup_run

_LOGGER = get_logger()


def _write_results(layout, values):
    """Write the metric values to results.csv in the eval folder."""
    path = os.path.join(layout.root, "results.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for name, value in values.items():
            writer.writerow([f"test_avg_{name}", value])
    _LOGGER.info("Wrote results to {}", path)


def run(config_path):
    """Evaluate the model named in the config and save the results."""
    config, layout, _ = setup_run(config_path, "eval")
    trainer = PipelineBuilder(config, layout).build()
    values = trainer.final_evaluate()
    _write_results(layout, values)


def main():
    """Console entry point for the evalsjepa command."""
    config_path = parse_config_arg("Evaluate the S-JEPA speech model")
    run(config_path)


if __name__ == "__main__":
    main()
