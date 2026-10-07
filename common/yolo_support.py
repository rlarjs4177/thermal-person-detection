from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from common.thermal import ThermalModule, enhance_feature


class JsonPersonDataset(YOLODataset):
    """Ultralytics dataset using the read-only source JSON directly."""
    def get_labels(self):
        labels = []
        for image_name in self.im_files:
            image = Path(image_name)
            ann_path = image.parent.parent / "gt" / f"{image.stem}.json"
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            width, height = float(ann["original_width"]), float(ann["original_height"])
            boxes = []
            for obj in ann.get("objects", []):
                if obj["label"].lower() != "person":
                    raise ValueError(f"unexpected class in {ann_path}")
                x, y, w, h = (float(obj[k]) for k in ("x", "y", "width", "height"))
                boxes.append([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])
            labels.append({"im_file": str(image), "shape": (int(height), int(width)),
                           "cls": np.zeros((len(boxes), 1), dtype=np.float32),
                           "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                           "segments": [], "keypoints": None, "normalized": True, "bbox_format": "xywh"})
        return labels


class JsonDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return JsonPersonDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=mode == "train", hyp=copy(self.args), rect=self.args.rect or mode == "val",
            cache=self.args.cache or None, single_cls=True, stride=gs, pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "), task="detect", classes=None, data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0)


class ThermalDetectionModel(DetectionModel):
    """Official Ultralytics graph plus Eq. (9) immediately before its Detect module."""
    def __init__(self, cfg, ch=3, nc=None, verbose=True, thermal_mode="full"):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.thermal_module = ThermalModule(thermal_mode)
        detects = [module for module in self.modules() if isinstance(module, Detect)]
        if len(detects) != 1:
            raise RuntimeError(f"expected exactly one official Detect head, found {len(detects)}")
        self._thermal_weight = None
        detects[0].register_forward_pre_hook(self._enhance_detect_inputs)

    def _enhance_detect_inputs(self, _module, args):
        if self._thermal_weight is None:
            raise RuntimeError("Thermal weight was not prepared before Detect.")
        features = args[0]
        enhanced = [enhance_feature(feature, self._thermal_weight) for feature in features]
        self.thermal_feature_shapes = [tuple(feature.shape) for feature in features]
        return (enhanced, *args[1:])

    def forward(self, x, *args, **kwargs):
        # DetectionModel.__init__ performs an official dummy forward to infer strides
        # before this subclass can register its new module/hook.
        if not hasattr(self, "thermal_module"):
            return super().forward(x, *args, **kwargs)
        image = x["img"] if isinstance(x, dict) else x
        self._thermal_weight = self.thermal_module(image)
        return super().forward(x, *args, **kwargs)


class ThermalDetectionTrainer(JsonDetectionTrainer):
    thermal_mode = "full"
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ThermalDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
            thermal_mode=self.thermal_mode))
        if weights:
            before = {k for k in model.state_dict() if not k.startswith("thermal_module.")}
            model.load(weights)
            official_keys = set(weights.float().state_dict() if hasattr(weights, "float") else weights.state_dict())
            missing = sorted(k for k in before if k not in official_keys)
            if missing:
                raise RuntimeError(f"unexpected missing pretrained detector keys: {missing[:20]}")
        return model
