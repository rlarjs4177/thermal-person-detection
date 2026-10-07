from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root(file: str) -> Path:
    root = Path(file).resolve().parent.parent
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    return root


def _metadata(root: Path, folder: Path) -> Path:
    from common.data import write_coco_metadata
    generated = folder / "generated"
    write_coco_metadata(root / "data", generated)
    return generated


def train_rfdetr(file: str, thermal: bool) -> None:
    root = _root(file)
    from rfdetr import RFDETRNano
    from common.rfdetr_support import install_rfdetr_training_patch
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0)
    if thermal: parser.add_argument("--thermal-mode", choices=("heatcore", "heatcore_diffusion", "full"), default="full")
    args = parser.parse_args(); mode = args.thermal_mode if thermal else None
    folder = Path(file).resolve().parent; dataset = _metadata(root, folder)
    output = folder / "outputs" / ((mode + "/") if mode else "") / f"seed_{args.seed}"
    install_rfdetr_training_patch(mode)
    model = RFDETRNano()
    model.train(dataset_dir=str(dataset), output_dir=str(output), resolution=384, epochs=1000,
        batch_size=16, grad_accum_steps=1, optimizer="adamw", lr=1e-4, lr_encoder=1e-4,
        weight_decay=1e-4, early_stopping=True, early_stopping_patience=100,
        early_stopping_min_delta=0.0, checkpoint_interval=100, seed=args.seed,
        multi_scale=False, scale_jitter=False, aug_config={"RandomHorizontalFlip": {"p": 0.5}},
        use_ema=False, run_test=True, notes={"thermal_mode": mode, "lambda_init": 1.0 if thermal else None})


def profile_rfdetr(file: str, thermal: bool) -> None:
    root = _root(file)
    import torch
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    from common.rfdetr_support import attach_rfdetr_thermal
    from common.runtime import parameter_counts
    parser = argparse.ArgumentParser(); parser.add_argument("--thermal-mode", choices=("heatcore", "heatcore_diffusion", "full"), default="full")
    args = parser.parse_args(); detector = RFDETRNano(device="cpu")
    model = detector.model.model
    if thermal: attach_rfdetr_thermal(model, args.thermal_mode)
    model.eval(); x = torch.rand(1, 3, 384, 384); mean=x.new_tensor((.485,.456,.406))[:,None,None]; std=x.new_tensor((.229,.224,.225))[:,None,None]
    with torch.no_grad(): out = model(nested_tensor_from_tensor_list([(x[0]-mean)/std]))
    total, trainable = parameter_counts(model)
    result = {"variant":"RF-DETR Nano", "input_resolution":384, "total_parameters":total,
              "trainable_parameters":trainable, "pred_logits":list(out["pred_logits"].shape), "pred_boxes":list(out["pred_boxes"].shape)}
    print(json.dumps(result, indent=2))
    if thermal:
        d=model.thermal_module.last_debug; print({k:tuple(v.shape) for k,v in d.items() if hasattr(v,"shape")},
            "W_range",(float(d["W"].min()),float(d["W"].max())),"scalars",model.thermal_module.learned_values(),
            "lambda",float(model.thermal_lambda.detach()),"optimizer_parameter_names",
            [n for n,p in model.named_parameters() if p.requires_grad and ("thermal_module" in n or n=="thermal_lambda")])
        attn=next(m for m in model.modules() if hasattr(m,"thermal_debug")); ad=attn.thermal_debug
        loc=ad["sampling_locations"]
        print({k:(tuple(v.shape) if hasattr(v,"shape") else v) for k,v in ad.items()},
            "sampling_coordinate_range",(float(loc.min()),float(loc.max())),
            "heads_levels_points",(loc.shape[2],loc.shape[3],loc.shape[4]))


def evaluate_rfdetr(file: str, thermal: bool) -> None:
    root = _root(file)
    from rfdetr import RFDETRNano
    from common.rfdetr_support import attach_rfdetr_thermal, rfdetr_evaluation_thermal_patch
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--thermal-mode",choices=("heatcore","heatcore_diffusion","full"),default="full")
    args=parser.parse_args(); folder=Path(file).resolve().parent; dataset=_metadata(root,folder)
    model=RFDETRNano(pretrain_weights=None,num_classes=1)
    if thermal: attach_rfdetr_thermal(model.model.model,args.thermal_mode)
    payload=__import__("torch").load(args.checkpoint,map_location="cpu",weights_only=False)
    state=payload.get("model",payload.get("state_dict",payload))
    target=model.model.model; target_keys=set(target.state_dict())
    candidates=[state]
    for prefix in ("model.","model.model.","module."):
        candidates.append({k[len(prefix):] if k.startswith(prefix) else k:v for k,v in state.items()})
    state=max(candidates,key=lambda candidate:len(target_keys.intersection(candidate)))
    if thermal:
        incompatible=target.load_state_dict(state,strict=True)
        thermal_parameters=[name for name,_ in target.named_parameters() if name=="thermal_lambda" or name.startswith("thermal_module.")]
        print({"thermal_parameters":thermal_parameters,"thermal_missing_keys":incompatible.missing_keys,
               "thermal_unexpected_keys":incompatible.unexpected_keys})
        with rfdetr_evaluation_thermal_patch(args.thermal_mode) as evaluation_models:
            metrics=model.evaluate(dataset_dir=str(dataset),split="test",resolution=384,batch_size=1,device="cpu")
        evaluation_model=evaluation_models[-1]
        print({"thermal_forward_called":bool(evaluation_model.thermal_module.last_debug),
               "thermal_attention_called":any(hasattr(module,"thermal_debug") for module in evaluation_model.modules())})
    else:
        state={k:v for k,v in state.items() if k in target_keys}
        missing,unexpected=target.load_state_dict(state,strict=False)
        if missing or unexpected: raise RuntimeError(f"checkpoint mismatch missing={missing}, unexpected={unexpected}")
        metrics=model.evaluate(dataset_dir=str(dataset),split="test",resolution=384,batch_size=1,device="cpu")
    print(metrics)
    result_path=Path(args.checkpoint).resolve().parent.parent/"results.json"
    if result_path.is_file():
        from common.runtime import write_json
        payload=json.loads(result_path.read_text(encoding="utf-8"))
        aliases={"test_map50_95":("map","mAP","map50_95","mAP_50_95"),"test_map50":("map50","mAP_50"),
                 "precision":("precision",),"recall":("recall",)}
        for target,names in aliases.items():
            value=next((metrics[name] for name in names if name in metrics),None)
            if value is not None: payload[target]=float(value)
        payload["test_precision"]=payload.get("precision"); payload["test_recall"]=payload.get("recall")
        write_json(result_path,payload)
