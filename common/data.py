from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 5200, "val": 2000, "test": 276}


def project_root(file: str | Path) -> Path:
    path = Path(file).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "data").is_dir():
            return parent
    raise FileNotFoundError(f"could not find project root containing data/ from {path}")


def scan_split(data_root: Path, split: str) -> list[tuple[Path, Path, dict[str, Any]]]:
    images = sorted((data_root / split / "image").glob("*.jpg"))
    labels = data_root / split / "gt"
    rows = []
    for image in images:
        ann_path = labels / f"{image.stem}.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"missing annotation: {ann_path}")
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        if ann.get("file_name") != image.name:
            raise ValueError(f"filename mismatch in {ann_path}")
        for obj in ann.get("objects", []):
            if obj.get("label", "").lower() != "person":
                raise ValueError(f"non-person class in {ann_path}: {obj.get('label')}")
            for key in ("x", "y", "width", "height"):
                if key not in obj:
                    raise ValueError(f"missing bbox key {key!r} in {ann_path}")
        rows.append((image, ann_path, ann))
    return rows


def inspect_dataset(data_root: Path, strict_counts: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"classes": ["person"], "bbox_format": "xywh_absolute_top_left"}
    for split in SPLITS:
        rows = scan_split(data_root, split)
        if strict_counts and len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]}, found {len(rows)}")
        summary[split] = len(rows)
    return summary


def validate_white_hot(data_root: Path, samples_per_split: int = 3) -> list[dict[str, Any]]:
    results = []
    for split in SPLITS:
        for image_path, _, _ in scan_split(data_root, split)[:samples_per_split]:
            arr = np.asarray(Image.open(image_path))
            max_channel_diff = int(np.abs(arr.astype(np.int16) - arr[..., :1].astype(np.int16)).max())
            results.append({
                "split": split, "file": str(image_path), "shape": list(arr.shape),
                "dtype": str(arr.dtype), "min": int(arr.min()), "max": int(arr.max()),
                "max_rgb_channel_difference": max_channel_diff,
                "warning": max_channel_diff > 3,
            })
    return results


def write_coco_metadata(data_root: Path, generated_root: Path) -> dict[str, Path]:
    """Generate COCO JSON only; image paths remain absolute and data/ stays untouched."""
    generated_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in SPLITS:
        images, annotations, ann_id = [], [], 1
        for image_id, (image_path, _, source) in enumerate(scan_split(data_root, split), 1):
            width, height = int(source["original_width"]), int(source["original_height"])
            images.append({"id": image_id, "file_name": str(image_path.resolve()), "width": width, "height": height})
            for obj in source.get("objects", []):
                bbox = [float(obj[k]) for k in ("x", "y", "width", "height")]
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": 0,
                                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
                ann_id += 1
        payload = {"images": images, "annotations": annotations,
                   "categories": [{"id": 0, "name": "person", "supercategory": "person"}]}
        out_dir = generated_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "_annotations.coco.json"
        encoded = json.dumps(payload, ensure_ascii=False)
        out_path.write_text(encoded, encoding="utf-8")
        if split == "val":  # RF-DETR's public Roboflow loader calls this directory "valid".
            valid_dir = generated_root / "valid"
            valid_dir.mkdir(parents=True, exist_ok=True)
            (valid_dir / "_annotations.coco.json").write_text(encoded, encoding="utf-8")
        outputs[split] = out_path
    return outputs
