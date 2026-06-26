"""Shared helpers for every command line program.

All programs read a config with -c/--config, set up logging into the run folder,
log the full config for traceability, and save the used config next to the
outputs. Putting this here keeps each program short and consistent.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from ..config_schema import load_experiment_config, save_used_config
from ..logging import get_logger, log_config, setup_logging
from ..rundir import RunDirectoryManager


def parse_config_arg(description):
    """Parse the -c/--config argument and return the config path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", required=True,
                        help="path to the YAML configuration file")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        parser.error(f"config file not found: {args.config}")
    return args.config


def _log_file(layout, kind):
    """Return a timestamped log file path inside the run folder."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(layout.logs_dir, f"{kind}_{stamp}.log")


def setup_run(config_path, kind):
    """Load the config, pick the run folder, and start logging.

    For a training run, the resume flag comes from `checkpoint.resume` in the
    config: when it is True and a usable checkpoint exists, the old run folder
    is reused. Evaluation always makes a fresh folder.

    Args:
        config_path: path to the YAML config.
        kind: "train" or "eval".

    Returns:
        A tuple (config, layout, resumed).
    """
    config = load_experiment_config(config_path)
    resume = kind == "train" and config.checkpoint.resume
    manager = RunDirectoryManager(config.runs_root, config.run_name, kind)
    layout, resumed = manager.resolve(resume=resume)
    setup_logging(level="DEBUG", logfile=_log_file(layout, kind))
    logger = get_logger()
    logger.info("Starting {} pipeline (run folder: {})", kind, layout.root)
    log_config("configuration", config.to_dict())
    save_used_config(config, layout.config_used)
    return config, layout, resumed
