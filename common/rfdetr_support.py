from __future__ import annotations

import types
import weakref
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path

from common.runtime import append_csv, parameter_counts, result_record, write_json

from common.thermal import ThermalModule, enhance_feature, sample_weight_at_locations


def _denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (tensor * std + mean).clamp(0.0, 1.0)


def _thermal_ms_deform_forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                               input_level_start_index, input_padding_mask=None, input_spatial_shapes_hw=None):
    """RF-DETR 1.9.0 eager MSDeformAttn with only the paper Eq. (10) insertion."""
    from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
    batch_size, len_query, _ = query.shape
    _, len_input, _ = input_flatten.shape
    expected = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()
    assert expected == len_input, "input_spatial_shapes must match the flattened input length"
    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], 0.0)
    logits = self.attention_weights(query).view(batch_size, len_query, self.n_heads, self.n_levels * self.n_points)
    offsets = self.sampling_offsets(query).view(batch_size, len_query, self.n_heads, self.n_levels, self.n_points, 2)
    if reference_points.shape[-1] == 2:
        normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        locations = reference_points[:, :, None, :, None, :2] + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError("reference_points last dimension must be 2 or 4")
    # --- Thermal cross-attention bias: Eq. (10), before official softmax ---
    owner = self._thermal_owner()
    sampled = sample_weight_at_locations(owner._thermal_weight, locations)
    logits = logits + owner.thermal_lambda * sampled.flatten(-2)
    self.thermal_debug = {"sampling_locations": locations.detach(), "attention_logits": logits.detach(),
                          "sampled_W": sampled.detach(), "bias_before_softmax": True}
    # --- End thermal cross-attention bias ---
    attention = F.softmax(logits, -1)
    value = value.transpose(1, 2).contiguous().view(batch_size, self.n_heads, self.d_model // self.n_heads, len_input)
    output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, locations, attention,
                                         value_spatial_shapes_hw=input_spatial_shapes_hw)
    return self.output_proj(output)


def attach_rfdetr_thermal(nn_model: nn.Module, mode: str) -> nn.Module:
    """Attach paper Eqs. (1)-(10) to an official RF-DETR Nano model in place."""
    from rfdetr.models.ops.modules import MSDeformAttn
    if hasattr(nn_model, "thermal_module"):
        return nn_model
    nn_model.add_module("thermal_module", ThermalModule(mode))
    nn_model.register_parameter("thermal_lambda", nn.Parameter(torch.tensor(1.0)))
    backbone = nn_model.backbone[0]
    original_forward = backbone.forward

    def thermal_backbone_forward(this, tensor_list):
        # RF-DETR sees ImageNet-normalized pixels; invert that exact affine transform to recover
        # the aligned post-spatial-transform [0,1] branch required by the paper.
        nn_model._thermal_weight = nn_model.thermal_module(_denormalize_imagenet(tensor_list.tensors))
        features, cross_features = original_forward(tensor_list)
        # --- Thermal module modification: Eq. (9), official projector outputs -> transformer ---
        for feature in features:
            feature.tensors = enhance_feature(feature.tensors, nn_model._thermal_weight)
        if cross_features is not None:
            for feature in cross_features:
                feature.tensors = enhance_feature(feature.tensors, nn_model._thermal_weight)
        # --- End thermal module modification ---
        return features, cross_features

    backbone.forward = types.MethodType(thermal_backbone_forward, backbone)
    owner_ref = weakref.ref(nn_model)
    patched = 0
    for module in nn_model.modules():
        if isinstance(module, MSDeformAttn):
            module._thermal_owner = owner_ref
            module.forward = types.MethodType(_thermal_ms_deform_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError("no official RF-DETR MSDeformAttn modules found")
    nn_model.thermal_attention_module_count = patched
    return nn_model


@contextmanager
def rfdetr_evaluation_thermal_patch(mode: str):
    """Attach the training-time thermal structure to RF-DETR's temporary evaluation model."""
    from rfdetr.training.module_model import RFDETRModelModule
    original = RFDETRModelModule.__init__
    evaluation_models = []

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        evaluation_models.append(attach_rfdetr_thermal(self.model, mode))

    RFDETRModelModule.__init__ = init
    try:
        yield evaluation_models
    finally:
        RFDETRModelModule.__init__ = original


class RequestedCheckpointCallback:
    """Factory-like callback implementation kept independent until Lightning is imported."""


def install_rfdetr_training_patch(mode: str | None) -> None:
    """Ensure the official Lightning module created by RFDETR.train receives the thermal patch."""
    import rfdetr.training as training_api
    from rfdetr.training.module_model import RFDETRModelModule
    from pytorch_lightning import Callback
    if getattr(RFDETRModelModule, "_thermal_patched", False):
        return
    original = RFDETRModelModule.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if mode is not None:
            attach_rfdetr_thermal(self.model, mode)
    RFDETRModelModule.__init__ = init
    RFDETRModelModule._thermal_patched = True

    original_builder = training_api.build_trainer
    class ArtifactCallback(Callback):
        def __init__(self, output_dir, seed):
            self.output_dir=Path(output_dir); self.seed=seed; self.best=-float("inf"); self.test_metrics={}
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking: return
            epoch=trainer.current_epoch+1; ckpt=self.output_dir/"checkpoints"; ckpt.mkdir(parents=True,exist_ok=True)
            metric=trainer.callback_metrics.get("val/mAP_50_95")
            value=float(metric.detach().cpu()) if metric is not None else float("nan")
            trainer.save_checkpoint(ckpt/"last.pt")
            if value>self.best: self.best=value; trainer.save_checkpoint(ckpt/"best.pt")
            if epoch%100==0: trainer.save_checkpoint(ckpt/f"epoch_{epoch:04d}.pt")
            model=pl_module.model; thermal=getattr(model,"thermal_module",None)
            vals=thermal.learned_values() if thermal else {"alpha_w1":None,"beta_w2":None,"gamma_w3":None}
            def scalar(value):
                return float(value.detach().cpu()) if hasattr(value,"detach") else value
            row={"epoch":epoch,"training_loss":scalar(trainer.callback_metrics.get("train/loss_epoch",trainer.callback_metrics.get("train/loss"))),
                "val_map50":trainer.callback_metrics.get("val/mAP_50"),"val_map50_95":metric,
                "precision":trainer.callback_metrics.get("val/precision"),"recall":trainer.callback_metrics.get("val/recall"),
                "learning_rate":trainer.optimizers[0].param_groups[0]["lr"],"alpha":vals["alpha_w1"],
                "beta":vals["beta_w2"],"gamma":vals["gamma_w3"],
                "lambda":None if not hasattr(model,"thermal_lambda") else float(model.thermal_lambda.detach().cpu())}
            for key,value in sorted(trainer.callback_metrics.items()):
                if key.startswith("train/") and (key.endswith("_epoch") or "loss" in key):
                    row["component_"+key.replace("/","_")]=scalar(value)
            append_csv(self.output_dir/"experiment_metrics.csv",row)
        def on_test_epoch_end(self, trainer, pl_module):
            aliases={"test_map50_95":"test/mAP_50_95","test_map50":"test/mAP_50",
                     "precision":"test/precision","recall":"test/recall"}
            for target,key in aliases.items():
                value=trainer.callback_metrics.get(key)
                if value is not None: self.test_metrics[target]=float(value.detach().cpu()) if hasattr(value,"detach") else float(value)
        def on_fit_end(self, trainer, pl_module):
            model=pl_module.model; thermal=getattr(model,"thermal_module",None)
            if thermal is not None:
                best_checkpoint=torch.load(self.output_dir/"checkpoints"/"best.pt",map_location="cpu",weights_only=False)
                pl_module.load_state_dict(best_checkpoint["state_dict"],strict=True)
                model=pl_module.model; thermal=model.thermal_module
            vals=thermal.learned_values() if thermal else {"alpha_w1":None,"beta_w2":None,"gamma_w3":None}
            total,trainable=parameter_counts(model)
            result=result_record(model="RF-DETR",variant="Nano",thermal_mode=mode,seed=self.seed,
                input_resolution=384,pretrained_checkpoint="rf-detr-nano.pth",
                total_parameters=total,trainable_parameters=trainable)
            result.update(best_val_map50_95=self.best,stopped_epoch=trainer.current_epoch+1,**vals,
                **{"lambda":None if not hasattr(model,"thermal_lambda") else float(model.thermal_lambda.detach().cpu())})
            result.update(self.test_metrics)
            result["test_precision"]=result.get("precision"); result["test_recall"]=result.get("recall")
            write_json(self.output_dir/"results.json",result)
    def build_trainer(config, model_config, **kwargs):
        trainer=original_builder(config,model_config,**kwargs)
        serialized=config.model_dump(mode="json") if hasattr(config,"model_dump") else vars(config)
        write_json(Path(config.output_dir)/"config.json",serialized)
        trainer.callbacks.append(ArtifactCallback(config.output_dir,config.seed))
        return trainer
    training_api.build_trainer=build_trainer
