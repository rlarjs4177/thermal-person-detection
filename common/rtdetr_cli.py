from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def setup(file):
    root=Path(file).resolve().parent.parent; vendor=root/"third_party"/"RT-DETR"/"rtdetr_pytorch"
    sys.path.insert(0,str(vendor)); sys.path.insert(0,str(root)); return root,vendor

def make_cfg(file, output, split="val", checkpoint=None):
    root,vendor=setup(file)
    from common.data import write_coco_metadata
    from common.rtdetr_support import configure_rtdetr, OFFICIAL_R34_CHECKPOINT
    from src.core import YAMLConfig
    folder=Path(file).resolve().parent; generated=folder/"generated"; write_coco_metadata(root/"data",generated)
    cfg=YAMLConfig(str(vendor/"configs"/"rtdetr"/"rtdetr_r34vd_6x_coco.yml"),tuning=checkpoint or OFFICIAL_R34_CHECKPOINT)
    configure_rtdetr(cfg,generated,output,split); return cfg,root

def train_rtdetr(file,thermal):
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int,default=0)
    if thermal: parser.add_argument("--thermal-mode",choices=("heatcore","heatcore_diffusion","full"),default="full")
    args=parser.parse_args(); mode=args.thermal_mode if thermal else None; folder=Path(file).resolve().parent
    output=folder/"outputs"/((mode+"/") if mode else "")/f"seed_{args.seed}"; cfg,_=make_cfg(file,output)
    from common.rtdetr_support import attach_rtdetr_thermal,ExperimentSolver
    from src.misc.dist import set_seed
    set_seed(args.seed); model=cfg.model
    if thermal: attach_rtdetr_thermal(model,mode)
    ExperimentSolver(cfg,args.seed,thermal).fit()
    best=output/"checkpoints"/"best.pt"
    if best.is_file(): _evaluate_rtdetr_checkpoint(file,best,"test",thermal,mode)

def _evaluate_rtdetr_checkpoint(file,checkpoint,split,thermal,mode):
    output=Path(checkpoint).resolve().parent.parent/"test_evaluation"; cfg,_=make_cfg(file,output,split,checkpoint=None)
    cfg.tuning=""; cfg.resume=str(checkpoint)
    from common.rtdetr_support import attach_rtdetr_thermal,ExperimentSolver
    if thermal: attach_rtdetr_thermal(cfg.model,mode)
    stats,evaluator=ExperimentSolver(cfg,0,thermal).val(); print(stats)
    result_path=Path(checkpoint).resolve().parent.parent/"results.json"
    if result_path.is_file():
        from common.runtime import write_json
        import numpy as np
        payload=json.loads(result_path.read_text(encoding="utf-8")); bbox=stats.get("coco_eval_bbox",[None]*12)
        key="test" if split=="test" else "best_val"
        payload[key+"_map50_95"]=None if bbox[0] is None else float(bbox[0]); payload[key+"_map50"]=None if bbox[1] is None else float(bbox[1])
        if split=="test" and evaluator is not None and "bbox" in evaluator.coco_eval and evaluator.coco_eval["bbox"].eval:
            e=evaluator.coco_eval["bbox"].eval; p=e.get("precision"); r=e.get("recall")
            payload["precision"]=payload["test_precision"]=float(p[p>=0].mean()) if p is not None and np.any(p>=0) else None
            payload["recall"]=payload["test_recall"]=float(r[r>=0].mean()) if r is not None and np.any(r>=0) else None
        write_json(result_path,payload)
    return stats,evaluator

def evaluate_rtdetr(file,thermal):
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--split",choices=("val","test"),default="test"); parser.add_argument("--thermal-mode",choices=("heatcore","heatcore_diffusion","full"),default="full")
    args=parser.parse_args()
    _evaluate_rtdetr_checkpoint(file,args.checkpoint,args.split,thermal,args.thermal_mode)

def profile_rtdetr(file,thermal):
    parser=argparse.ArgumentParser(); parser.add_argument("--thermal-mode",choices=("heatcore","heatcore_diffusion","full"),default="full")
    args=parser.parse_args(); folder=Path(file).resolve().parent; cfg,_=make_cfg(file,folder/"profile_output")
    from common.rtdetr_support import attach_rtdetr_thermal,ExperimentSolver
    from common.runtime import parameter_counts
    import torch
    if thermal: attach_rtdetr_thermal(cfg.model,args.thermal_mode)
    solver=ExperimentSolver(cfg,0,thermal)._solver; solver.setup(); model=solver.model.eval()
    with torch.no_grad(): out=model(torch.rand(1,3,640,640,device=solver.device))
    total,trainable=parameter_counts(model); print(json.dumps({"variant":"RT-DETR-R34","input_resolution":640,"total_parameters":total,"trainable_parameters":trainable,"pred_logits":list(out["pred_logits"].shape),"pred_boxes":list(out["pred_boxes"].shape)},indent=2))
    if thermal:
        raw=model.module if hasattr(model,"module") else model; d=raw.thermal_module.last_debug
        print({k:tuple(v.shape) for k,v in d.items() if hasattr(v,"shape")},"W_range",(float(d["W"].min()),float(d["W"].max())),
            "scalars",raw.thermal_module.learned_values(),"lambda",float(raw.thermal_lambda.detach()),"optimizer_parameter_names",
            [n for n,p in raw.named_parameters() if p.requires_grad and ("thermal_module" in n or n=="thermal_lambda")])
        attn=next(m for m in raw.modules() if hasattr(m,"thermal_debug")); ad=attn.thermal_debug; loc=ad["sampling_locations"]
        print({k:(tuple(v.shape) if hasattr(v,"shape") else v) for k,v in ad.items()},
            "sampling_coordinate_range",(float(loc.min()),float(loc.max())),"heads_levels_points",(loc.shape[2],loc.shape[3],loc.shape[4]))
