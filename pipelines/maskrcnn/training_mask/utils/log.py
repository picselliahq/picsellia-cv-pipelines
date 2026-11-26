"""Progress logging for the pipeline (stdlib only).

Records go to the *current* ``sys.stdout`` so they land in the step log files the
engine captures, with a timestamp and a level, instead of bare ``print`` calls.
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "maskrcnn_mask"


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that always writes to the current ``sys.stdout``.

    The engine may swap ``sys.stdout`` per step; a handler bound once at import
    time would keep writing to the old stream.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # noqa: ARG002 - always follow sys.stdout
        pass


def get_logger() -> logging.Logger:
    """Return the pipeline logger (configured once)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def fmt_duration(seconds: float) -> str:
    """Format a duration as ``42s``, ``3m05s`` or ``1h02m``."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
