from __future__ import annotations

import json
import time
import types
import weakref
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common.runtime import append_csv, parameter_counts, result_record, write_json
from common.thermal import ThermalModule, enhance_feature, sample_weight_at_locations


OFFICIAL_R34_CHECKPOINT = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"


def _thermal_rtdetr_attention(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
    """Original-author MSDeformableAttention plus Eq. (10), marked at insertion."""
    bs, len_q = query.shape[:2]; len_v = value.shape[1]
    value = self.value_proj(value)
    if value_mask is not None:
        value *= value_mask.to(value.dtype).unsqueeze(-1)
    value = value.reshape(bs, len_v, self.num_heads, self.head_dim)
    offsets = self.sampling_offsets(query).reshape(bs, len_q, self.num_heads, self.num_levels, self.num_points, 2)
    logits = self.attention_weights(query).reshape(bs, len_q, self.num_heads, self.num_levels, self.num_points)
    if reference_points.shape[-1] == 2:
        normalizer = torch.as_tensor(value_spatial_shapes, device=query.device, dtype=query.dtype).flip([1]).reshape(1,1,1,self.num_levels,1,2)
        locations = reference_points.reshape(bs,len_q,1,self.num_levels,1,2) + offsets / normalizer
    elif reference_points.shape[-1] == 4:
        locations = reference_points[:,:,None,:,None,:2] + offsets / self.num_points * reference_points[:,:,None,:,None,2:] * 0.5
    else:
        raise ValueError("reference_points last dimension must be 2 or 4")
    # --- Thermal cross-attention bias: Eq. (10), before official softmax ---
    owner = self._thermal_owner()
    sampled = sample_weight_at_locations(owner._thermal_weight, locations)
    logits = logits + owner.thermal_lambda * sampled
    self.thermal_debug = {"sampling_locations": locations.detach(), "attention_logits": logits.detach(),
                          "sampled_W": sampled.detach(), "bias_before_softmax": True}
    # --- End thermal cross-attention bias ---
    attention = F.softmax(logits.flatten(-2), dim=-1).reshape(bs,len_q,self.num_heads,self.num_levels,self.num_points)
    return self.output_proj(self.ms_deformable_attn_core(value, value_spatial_shapes, locations, attention))


def attach_rtdetr_thermal(model: nn.Module, mode: str) -> nn.Module:
    from src.zoo.rtdetr.rtdetr_decoder import MSDeformableAttention
    if hasattr(model, "thermal_module"): return model
    model.add_module("thermal_module", ThermalModule(mode))
    model.register_parameter("thermal_lambda", nn.Parameter(torch.tensor(1.0)))

    def forward(this, x, targets=None):
        if this.multi_scale and this.training:
            raise RuntimeError("multi-scale must be disabled so Thermal and detector branches stay aligned")
        this._thermal_weight = this.thermal_module(x)
        features = this.backbone(x)
        # --- Thermal module modification: Eq. (9), backbone output -> HybridEncoder ---
        features = [enhance_feature(feature, this._thermal_weight) for feature in features]
        # --- End thermal module modification ---
        return this.decoder(this.encoder(features), targets)
    model.forward = types.MethodType(forward, model)
    owner_ref=weakref.ref(model); patched=0
    for module in model.modules():
        if isinstance(module, MSDeformableAttention):
            module._thermal_owner=owner_ref
            module.forward=types.MethodType(_thermal_rtdetr_attention,module)
            patched += 1
    if patched == 0: raise RuntimeError("no official RT-DETR MSDeformableAttention found")
    model.thermal_attention_module_count=patched
    return model


def configure_rtdetr(cfg, generated: Path, output: Path, split: str = "val"):
    y=cfg.yaml_cfg
    output_dir=str(Path(output).resolve())
    y.update({"num_classes":1,"remap_mscoco_category":False,"epoches":1000,"checkpoint_step":100,
              "output_dir":output_dir,"use_ema":False,"sync_bn":False,"find_unused_parameters":False})
    cfg.epoches=1000
    cfg.output_dir=output_dir
    y["RTDETR"]["multi_scale"] = None
    y["optimizer"]={"type":"AdamW","lr":0.0001,"betas":[0.9,0.999],"weight_decay":0.0001}
    y["lr_scheduler"]={"type":"MultiStepLR","milestones":[1000],"gamma":0.1}
    train=y["train_dataloader"]; train["batch_size"]=16; train["num_workers"]=4
    train["dataset"]["img_folder"]=str(generated/"train"); train["dataset"]["ann_file"]=str(generated/"train"/"_annotations.coco.json")
    train["dataset"]["transforms"]["ops"]=[{"type":"RandomHorizontalFlip"},{"type":"Resize","size":[640,640]},
        {"type":"ToImageTensor"},{"type":"ConvertDtype"},{"type":"SanitizeBoundingBox","min_size":1},
        {"type":"ConvertBox","out_fmt":"cxcywh","normalize":True}]
    val=y["val_dataloader"]; val["batch_size"]=1; val["num_workers"]=2
    val["dataset"]["img_folder"]=str(generated/split); val["dataset"]["ann_file"]=str(generated/split/"_annotations.coco.json")
    return cfg


class ExperimentSolver:
    """Adapter over the official DetSolver with requested checkpointing and early stopping."""
    def __init__(self, cfg, seed: int, thermal: bool):
        from src.solver import DetSolver
        self._solver=DetSolver(cfg); self.cfg=cfg; self.seed=seed; self.thermal=thermal

    def fit(self):
        from src.misc import dist
        from src.data import get_coco_api_from_dataset
        from src.solver.det_engine import train_one_epoch, evaluate
        s=self._solver; s.train(); base_ds=get_coco_api_from_dataset(s.val_dataloader.dataset)
        best=-float("inf"); best_map50=None; best_epoch=None; counter=0; out=Path(s.output_dir); (out/"checkpoints").mkdir(exist_ok=True)
        if dist.is_main_process():
            write_json(out/"config.json",{"seed":self.seed,"epochs":1000,"batch_size":16,"optimizer":"AdamW",
                "initial_learning_rate":1e-4,"weight_decay":1e-4,"early_stopping_monitor":"val_map50_95",
                "early_stopping_patience":100,"early_stopping_min_delta":0.0,"input_resolution":640})
        for epoch in range(s.last_epoch+1, self.cfg.epoches):
            stats=train_one_epoch(s.model,s.criterion,s.train_dataloader,s.optimizer,s.device,epoch,
                self.cfg.clip_max_norm,print_freq=self.cfg.log_step,ema=s.ema,scaler=s.scaler)
            s.lr_scheduler.step(); module=s.ema.module if s.ema else s.model
            val,coco=evaluate(module,s.criterion,s.postprocessor,s.val_dataloader,base_ds,s.device,s.output_dir)
            bbox=val.get("coco_eval_bbox",[float("nan")]*12); metric=float(bbox[0]); improved=metric>best
            if improved: best=metric; best_map50=float(bbox[1]); best_epoch=epoch+1; counter=0
            else: counter+=1
            model=dist.de_parallel(s.model); thermal_module=getattr(model,"thermal_module",None)
            state=s.state_dict(epoch); state.update(best_validation_metric=best,early_stopping_counter=counter,
                random_seed=self.seed,training_config=self.cfg.yaml_cfg,
                thermal_parameters=None if thermal_module is None else thermal_module.learned_values(),
                thermal_lambda=None if not hasattr(model,"thermal_lambda") else float(model.thermal_lambda.detach().cpu()))
            dist.save_on_master(state,out/"checkpoints"/"last.pt")
            if improved: dist.save_on_master(state,out/"checkpoints"/"best.pt")
            if (epoch+1)%100==0: dist.save_on_master(state,out/"checkpoints"/f"epoch_{epoch+1:04d}.pt")
            precision=recall=None
            if coco is not None and "bbox" in coco.coco_eval and coco.coco_eval["bbox"].eval:
                e=coco.coco_eval["bbox"].eval; p=e.get("precision"); r=e.get("recall")
                if p is not None and np.any(p>=0): precision=float(p[p>=0].mean())
                if r is not None and np.any(r>=0): recall=float(r[r>=0].mean())
            values=thermal_module.learned_values() if thermal_module else {"alpha_w1":None,"beta_w2":None,"gamma_w3":None}
            row={"epoch":epoch+1,"training_loss":stats.get("loss"),"val_map50":float(bbox[1]),"val_map50_95":metric,
                 "precision":precision,"recall":recall,"learning_rate":s.optimizer.param_groups[0]["lr"],
                 "alpha":values["alpha_w1"],"beta":values["beta_w2"],"gamma":values["gamma_w3"],
                 "lambda":state["thermal_lambda"]}
            for k,v in stats.items(): row[f"train_{k}"]=v
            append_csv(out/"metrics.csv",row)
            if counter>=100: break
        model=dist.de_parallel(s.model); total,trainable=parameter_counts(model)
        thermal_module=getattr(model,"thermal_module",None)
        if thermal_module is not None:
            best_checkpoint=torch.load(out/"checkpoints"/"best.pt",map_location="cpu",weights_only=False)
            model.load_state_dict(best_checkpoint["model"],strict=True)
            thermal_module=model.thermal_module
        values=thermal_module.learned_values() if thermal_module else {"alpha_w1":None,"beta_w2":None,"gamma_w3":None}
        result=result_record(model="RT-DETR",variant="R34",thermal_mode=(thermal_module.mode if thermal_module else None),
            seed=self.seed,input_resolution=640,pretrained_checkpoint="rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
            total_parameters=total,trainable_parameters=trainable)
        result.update(best_val_map50_95=best,best_val_map50=best_map50,best_epoch=best_epoch,stopped_epoch=epoch+1,**values,
            **{"lambda":None if not hasattr(model,"thermal_lambda") else float(model.thermal_lambda.detach().cpu())})
        if dist.is_main_process(): write_json(out/"results.json",result)
        return result

    def val(self):
        from src.data import get_coco_api_from_dataset
        from src.solver.det_engine import evaluate
        s=self._solver; s.eval(); base_ds=get_coco_api_from_dataset(s.val_dataloader.dataset)
        module=s.ema.module if s.ema else s.model
        return evaluate(module,s.criterion,s.postprocessor,s.val_dataloader,base_ds,s.device,s.output_dir)
