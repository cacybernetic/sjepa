"""Record the training history to a CSV file and keep it in memory.

After every epoch we append one row with the train and validation values. The
CSV file lets us reopen the run later, and the in-memory list feeds the history
plots. The header is written once and grows to fit any new metric name.
"""

from __future__ import annotations

import csv
import os

from .logging import get_logger

_LOGGER = get_logger()


class HistoryRecorder:
    """Append epoch rows to a CSV file and remember them in memory."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.rows = []
        self._load_existing()

    def _load_existing(self):
        """Read an existing CSV so a resumed run keeps its old history."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.rows.append(self._cast_row(row))
        _LOGGER.info("Loaded {} history rows from {}", len(self.rows),
                     self.csv_path)

    @staticmethod
    def _cast_row(row):
        """Turn CSV text values into numbers when possible."""
        out = {}
        for key, value in row.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = value
        if "epoch" in out:
            out["epoch"] = int(out["epoch"])
        return out

    def _all_fields(self):
        """Return the union of every column name, with epoch first."""
        names = set()
        for row in self.rows:
            names.update(row.keys())
        names.discard("epoch")
        return ["epoch"] + sorted(names)

    def _rewrite(self):
        """Write the whole CSV file from the in-memory rows."""
        fields = self._all_fields()
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def add(self, row):
        """Add one epoch row and rewrite the CSV file."""
        self.rows.append(dict(row))
        self._rewrite()
        return row

    def history(self):
        """Return the list of rows for plotting."""
        return self.rows
