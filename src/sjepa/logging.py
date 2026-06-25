"""Logging and terminal styling (loguru based, tqdm friendly).

This module centralizes terminal output. Rules enforced by the project spec:

    * use loguru for journaling, WITHOUT retention (no rotating files kept);
    * geeky but clean terminal rendering: ANSI colors allowed, but no emoji and
      no special characters that are not typeable on an English keyboard;
    * never break the tqdm progress bars: log records are written through
      ``tqdm.write`` so they appear above the live bars instead of corrupting
      them.

The public entry point is ``setup_logging`` which (re)configures the global
loguru logger. Helper functions provide a small ANSI palette for hand-rolled
status lines (model summaries, banners, etc.).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

from loguru import logger
from tqdm import tqdm

# Matches ANSI SGR escape sequences (colors/styles) to strip them from files.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# ---------------------------------------------------------------------------
# ANSI palette (plain escape codes, no external dependency).
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
_COLORS = {
    "black": "\033[30m", "red": "\033[31m", "green": "\033[32m",
    "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m",
    "cyan": "\033[36m", "white": "\033[37m", "grey": "\033[90m",
}


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Wrap ``text`` in an ANSI color (and optional bold) escape sequence."""
    code = _COLORS.get(color, "")
    prefix = (BOLD if bold else "") + code
    return f"{prefix}{text}{RESET}"


def _tqdm_sink(message: str) -> None:
    """Loguru sink that routes records through tqdm to preserve progress bars."""
    tqdm.write(message, end="")


# Loguru format: ``HH:MM:SS LEVEL    | message`` (matches the spec preview).
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level>| "
    "{message}"
)

_CONFIGURED = False


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    """Configure the global loguru logger.

    Args:
        level: minimum level for the console sink ("DEBUG", "INFO", ...).
        logfile: optional path to a plain-text log file. No retention/rotation
            policy is attached (per spec: journaling without retention).
    """
    global _CONFIGURED
    logger.remove()
    # Console sink through tqdm so live bars are never corrupted.
    logger.add(
        _tqdm_sink,
        level='DEBUG',
        format=_LOG_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    if logfile is not None:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)) or ".",
                    exist_ok=True)
        # File sink without retention/rotation. We strip ANSI escape sequences
        # (our banners/colorize embed raw ANSI in the message) so the log file
        # stays plain text.
        handle = open(logfile, "a", encoding="utf-8")

        def _file_sink(message: str) -> None:
            handle.write(_ANSI_RE.sub("", message))
            handle.flush()

        logger.add(
            _file_sink,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} | {message}",
            colorize=False,
            backtrace=False,
            diagnose=False,
            enqueue=False,
        )
    _CONFIGURED = True


def get_logger():
    """Return the shared loguru logger, configuring it lazily if needed."""
    if not _CONFIGURED:
        setup_logging()
    return logger


def banner(title: str, color: str = "cyan") -> str:
    """Build a simple ASCII banner line, e.g. ``===== title =====``."""
    return colorize(f"===== {title} =====", color, bold=True)


def log_hparams(title: str, items: dict, color: str = "cyan") -> None:
    """Print a small table of names and values for one component.

    This helps traceability: every component can show the settings it uses
    when it is built. The output stays plain and easy to read.

    Args:
        title: a short title for the block of values.
        items: a dict that maps a setting name to its value.
        color: the color used for the title banner.
    """
    log = get_logger()
    log.info(banner(title, color=color))
    for key, value in items.items():
        log.info("  {:<28} = {}", key, value)
