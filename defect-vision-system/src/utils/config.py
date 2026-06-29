"""Utility helpers: config loading and logging setup."""
import logging
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the YAML project configuration file."""
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and models/logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    log_dir = PROJECT_ROOT / "models" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
