from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def result_record(*, model: str, variant: str, thermal_mode: str | None, seed: int,
                  input_resolution: int, pretrained_checkpoint: str,
                  total_parameters: int | None = None,
                  trainable_parameters: int | None = None) -> dict[str, Any]:
    """Stable result schema shared by all eight experiments.

    Metrics that do not exist until a complete training/evaluation run intentionally
    remain null instead of being guessed from a smoke test.
    """
    return {
        "model": model,
        "variant": variant,
        "thermal_mode": thermal_mode,
        "seed": seed,
        "input_resolution": input_resolution,
        "pretrained_checkpoint": pretrained_checkpoint,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_epoch": None,
        "stopped_epoch": None,
        "best_val_map50": None,
        "best_val_map50_95": None,
        "test_map50": None,
        "test_map50_95": None,
        "precision": None,
        "recall": None,
        "test_precision": None,
        "test_recall": None,
        "alpha_w1": None,
        "beta_w2": None,
        "gamma_w3": None,
        "lambda": None,
    }
