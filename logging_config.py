"""Central logging setup shared by the CLI, the web app and the tests."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Configure and return the application logger.

    Safe to call more than once: handlers are attached only on the first call.
    """
    logger = logging.getLogger("ArchieAgent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "archie.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
