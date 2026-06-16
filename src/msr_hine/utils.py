"""Utility functions: seeding, device selection, logging, Hydra config helpers."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Also sets PYTHONHASHSEED and configures cuDNN to deterministic mode
    (may slow down training slightly).

    Args:
        seed: Integer seed value; logged at INFO level.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info("Global seed set to %d", seed)


def get_device(cfg_device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        cfg_device: "auto" (pick CUDA if available, else CPU), "cuda", "cpu",
                    or a specific "cuda:N".

    Returns:
        Resolved torch.device.
    """
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    log.info("Using device: %s", device)
    return device


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a compact formatter.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", ...).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=numeric_level,
    )


def cfg_to_dataclass(cfg: DictConfig, cls: type[T]) -> T:
    """Convert a Hydra DictConfig node to a Python dataclass instance.

    Performs a strict field-by-field conversion so that unknown keys in
    the config raise an error rather than being silently ignored.

    Args:
        cfg: Hydra config node.
        cls: Target dataclass type.

    Returns:
        Populated dataclass instance.

    Raises:
        TypeError: If cfg contains keys not present in cls.
        KeyError: If required dataclass fields are missing from cfg.
    """
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(cfg_dict) - field_names
    if unknown:
        raise TypeError(f"Unknown keys for {cls.__name__}: {unknown}")
    return cls(**cfg_dict)


@dataclass
class ExperimentConfig:
    """Flat config dataclass for a single experiment run.

    Populated from the resolved Hydra config by cfg_to_dataclass.
    """

    seed: int = 42
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "outputs"
