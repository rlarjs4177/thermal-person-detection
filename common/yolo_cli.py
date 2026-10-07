from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def prepare_env(root: Path) -> None:
    settings = root / ".ultralytics"
    settings.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))


def train_yolo(file: str, checkpoint: str, thermal: bool) -> None:
    root = Path(file).resolve().parent.parent
    prepare_env(root)
    from ultralytics import YOLO
    from common.yolo_support import JsonDetectionTrainer, ThermalDetectionTrainer
    from common.runtime import append_csv, parameter_counts, result_record, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    if thermal:
        parser.add_argument("--thermal-mode", choices=("heatcore", "heatcore_diffusion", "full"), default="full")
    args = parser.parse_args()
    mode = args.thermal_mode if thermal else None
    run_dir = Path(file).resolve().parent / "outputs" / ((mode + "/") if mode else "") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer_cls = ThermalDetectionTrainer if thermal else JsonDetectionTrainer
    if thermal:
        trainer_cls.thermal_mode = mode

    model = YOLO(checkpoint)

    def epoch_log(trainer):
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        row = {"epoch": trainer.epoch + 1, "training_loss": float(trainer.loss.detach().cpu()) if trainer.loss is not None else None,
               "val_map50": trainer.metrics.get("metrics/mAP50(B)"),
               "val_map50_95": trainer.metrics.get("metrics/mAP50-95(B)"),
               "precision": trainer.metrics.get("metrics/precision(B)"), "recall": trainer.metrics.get("metrics/recall(B)"),
               "learning_rate": trainer.lr.get("lr/pg0") if isinstance(trainer.lr, dict) else None,
               "alpha": None, "beta": None, "gamma": None, "lambda": None}
        try:
            row.update({f"component_{k}": float(v) for k,v in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
        except (AttributeError, TypeError, ValueError):
            pass
        if thermal:
            vals = raw_model.thermal_module.learned_values()
            row.update(alpha=vals["alpha_w1"], beta=vals["beta_w2"], gamma=vals["gamma_w3"])
        append_csv(run_dir / "metrics.csv", row)

    def mirror_checkpoints(trainer):
        ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
        if Path(trainer.last).exists(): shutil.copy2(trainer.last, ckpt_dir / "last.pt")
        if Path(trainer.best).exists(): shutil.copy2(trainer.best, ckpt_dir / "best.pt")
        epoch = trainer.epoch + 1
        if epoch % 100 == 0 and Path(trainer.last).exists():
            shutil.copy2(trainer.last, ckpt_dir / f"epoch_{epoch:04d}.pt")

    def finish(trainer):
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        total, trainable = parameter_counts(raw_model)
        if thermal:
            best_checkpoint = run_dir / "checkpoints" / "best.pt"
            if not best_checkpoint.is_file():
                raise FileNotFoundError(f"best checkpoint not found: {best_checkpoint}")
            best_model = YOLO(str(best_checkpoint)).model
            vals = best_model.thermal_module.learned_values()
        else:
            vals = {"alpha_w1": None, "beta_w2": None, "gamma_w3": None}
        result = result_record(model=checkpoint.removesuffix(".pt"), variant=checkpoint,
            thermal_mode=mode, seed=args.seed, input_resolution=640,
            pretrained_checkpoint=checkpoint, total_parameters=total, trainable_parameters=trainable)
        result.update(best_epoch=getattr(trainer.stopper, "best_epoch", None),
            stopped_epoch=trainer.epoch + 1, best_val_map50_95=getattr(trainer, "best_fitness", None),
            alpha_w1=vals["alpha_w1"], beta_w2=vals["beta_w2"], gamma_w3=vals["gamma_w3"])
        write_json(run_dir / "results.json", result)
        write_json(run_dir / "config.json", {**vars(args),"epochs":1000,"batch_size":16,"optimizer":"AdamW",
            "initial_learning_rate":1e-4,"weight_decay":1e-4,"early_stopping_monitor":"val_map50_95",
            "early_stopping_mode":"max","early_stopping_patience":100,"early_stopping_min_delta":0.0,
            "input_resolution":640,"augmentations":{"horizontal_flip":True,"hue":False,"saturation":False,
            "brightness":False,"contrast":False}})

    model.add_callback("on_fit_epoch_end", epoch_log)
    model.add_callback("on_model_save", mirror_checkpoints)
    model.add_callback("on_train_end", finish)
    model.train(trainer=trainer_cls, data=str(Path(file).resolve().parent / "config.yaml"), imgsz=640,
        epochs=1000, batch=16, optimizer="AdamW", lr0=1e-4, weight_decay=1e-4, patience=100,
        save=True, save_period=100, seed=args.seed, deterministic=True, single_cls=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, close_mosaic=0,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    # Final test is always the independently saved best validation checkpoint.
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    if best_checkpoint.is_file():
        _evaluate_yolo_checkpoint(Path(file), best_checkpoint, "test")


def _evaluate_yolo_checkpoint(file: Path, checkpoint: Path, split: str):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from common.yolo_support import JsonPersonDataset
    from ultralytics.utils.torch_utils import unwrap_model
    from copy import copy
    from ultralytics.utils import colorstr

    class JsonValidator(DetectionValidator):
        def build_dataset(self, img_path, mode="val", batch=None):
            gs = max(int(self.stride), 32)
            return JsonPersonDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                augment=False, hyp=copy(self.args), rect=True, cache=None, single_cls=True, stride=gs,
                pad=0.5, prefix=colorstr(f"{mode}: "), task="detect", classes=None, data=self.data, fraction=1.0)

    metrics = YOLO(str(checkpoint)).val(validator=JsonValidator,
        data=str(file.resolve().parent / "config.yaml"), split=split, imgsz=640, batch=1)
    result_path = checkpoint.resolve().parent.parent / "results.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        prefix = "test_" if split == "test" else "best_val_"
        payload[prefix + "map50"] = float(metrics.box.map50)
        payload[prefix + "map50_95"] = float(metrics.box.map)
        if split == "test":
            payload["precision"] = payload["test_precision"] = float(metrics.box.mp)
            payload["recall"] = payload["test_recall"] = float(metrics.box.mr)
        from common.runtime import write_json
        write_json(result_path, payload)
    return metrics


def evaluate_yolo(file: str, thermal: bool) -> None:
    root = Path(file).resolve().parent.parent; prepare_env(root)
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    _evaluate_yolo_checkpoint(Path(file), Path(args.checkpoint), args.split)


def profile_yolo(file: str, checkpoint: str, thermal: bool) -> None:
    root = Path(file).resolve().parent.parent; prepare_env(root)
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from common.yolo_support import ThermalDetectionModel
    from common.runtime import parameter_counts
    parser = argparse.ArgumentParser(); parser.add_argument("--thermal-mode", choices=("heatcore", "heatcore_diffusion", "full"), default="full")
    args = parser.parse_args()
    official = YOLO(checkpoint).model
    cls = ThermalDetectionModel if thermal else DetectionModel
    kwargs = {"thermal_mode": args.thermal_mode} if thermal else {}
    model = cls(official.yaml, nc=1, verbose=False, **kwargs); model.load(official); model.eval()
    image = torch.rand(1, 3, 640, 640)
    with torch.no_grad(): output = model(image)
    total, trainable = parameter_counts(model)
    print(json.dumps({"variant": checkpoint, "input_resolution": 640, "total_parameters": total,
                      "trainable_parameters": trainable, "forward_output_type": type(output).__name__}, indent=2))
    if thermal:
        d = model.thermal_module.last_debug
        print({k: tuple(v.shape) for k, v in d.items() if hasattr(v, "shape")},
              "W_range", (float(d["W"].min()), float(d["W"].max())),
              "detect_input_feature_shapes", model.thermal_feature_shapes,
              "learned_scalars", model.thermal_module.learned_values(),
              "optimizer_parameter_names", [n for n,p in model.named_parameters() if p.requires_grad and "thermal_module" in n])
