"""Profile the eight trained detectors without invoking training or evaluation code.

Only temporary files and ``profile_results.csv`` are written. In particular, the
evaluation entry points are deliberately not used because they traverse the test
set and may update a run's results.json.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ENV_YOLO = "geon_thermal_yolo"
ENV_RFDETR = "geon_thermal_rfdetr"
ENV_RTDETR = "geon_thermal_rtdetr"

# Change this one value to profile another explicitly selected seed. A missing
# checkpoint is an error; profile_all.py never falls back to a different seed.
PROFILE_SEED = 0

WARMUP = 30
REPEAT = 100

RUNS = (
    {"name": "YOLOv8m Baseline", "family": "yolo", "env": ENV_YOLO,
     "checkpoint": "yolo8_baseline/outputs/seed_{seed}/weights/best.pt", "size": 640, "thermal": False},
    {"name": "YOLOv8m Full", "family": "yolo", "env": ENV_YOLO,
     "checkpoint": "yolo8_thermal/outputs/full/seed_{seed}/weights/best.pt", "size": 640, "thermal": True},
    {"name": "YOLOv11m Baseline", "family": "yolo", "env": ENV_YOLO,
     "checkpoint": "yolo11_baseline/outputs/seed_{seed}/weights/best.pt", "size": 640, "thermal": False},
    {"name": "YOLOv11m Full", "family": "yolo", "env": ENV_YOLO,
     "checkpoint": "yolo11_thermal/outputs/full/seed_{seed}/weights/best.pt", "size": 640, "thermal": True},
    {"name": "RF-DETR Nano Baseline", "family": "rfdetr", "env": ENV_RFDETR,
     "checkpoint": "rfdetr_baseline/outputs/seed_{seed}/checkpoints/best.pt", "size": 384, "thermal": False},
    {"name": "RF-DETR Nano Full", "family": "rfdetr", "env": ENV_RFDETR,
     "checkpoint": "rfdetr_thermal/outputs/full/seed_{seed}/checkpoints/best.pt", "size": 384, "thermal": True},
    {"name": "RT-DETR-R34 Baseline", "family": "rtdetr", "env": ENV_RTDETR,
     "checkpoint": "rtdetr_baseline/outputs/seed_{seed}/checkpoints/best.pt", "size": 640, "thermal": False},
    {"name": "RT-DETR-R34 Full", "family": "rtdetr", "env": ENV_RTDETR,
     "checkpoint": "rtdetr_thermal/outputs/full/seed_{seed}/checkpoints/best.pt", "size": 640, "thermal": True},
)


_CONDA_ENV_CACHE: dict[str, Path] | None = None


def get_conda_envs() -> dict[str, Path]:
    global _CONDA_ENV_CACHE
    if _CONDA_ENV_CACHE is not None:
        return _CONDA_ENV_CACHE
    conda_candidates = (
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "anaconda3" / "Scripts" / "conda.exe"),
        str(Path.home() / "miniconda3" / "Scripts" / "conda.exe"),
    )
    conda_exe = next((value for value in conda_candidates if value and Path(value).is_file()), None)
    if conda_exe is None:
        raise RuntimeError("conda를 찾을 수 없습니다. Conda가 등록된 PowerShell에서 실행하세요.")
    completed = subprocess.run(
        [conda_exe, "env", "list", "--json"], capture_output=True, text=True, check=True
    )
    envs = {}
    for value in json.loads(completed.stdout).get("envs", []):
        path = Path(value)
        envs[path.name] = path
    _CONDA_ENV_CACHE = envs
    return envs


def get_env_python(name: str) -> Path:
    envs = get_conda_envs()
    if name not in envs:
        listing = "\n".join(f"  {key}: {value}" for key, value in sorted(envs.items()))
        raise RuntimeError(f"Conda 환경을 찾을 수 없습니다: {name}\n{listing}")
    python = envs[name] / ("python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"Conda Python을 찾을 수 없습니다: {python}")
    return python


# This code runs inside the selected model's own Conda environment. Explicit
# loaders replace the old global nn.Module._call_impl monkey patch: Ultralytics
# wrappers may call an inner model's forward method directly, so intercepting
# _call_impl did not reliably observe YOLO11's detector call.
WORKER_CODE = r"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ["PROFILE_ROOT"])
sys.path.insert(0, str(ROOT))

CONFIG_FILE = Path(sys.argv[1])
RESULT_FILE = Path(sys.argv[2])
config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def checkpoint_state(payload):
    if not isinstance(payload, dict):
        return payload
    return payload.get("model", payload.get("state_dict", payload))


def align_state_dict(state, model):
    # Choose only a prefix-normalized view; never discard checkpoint keys.
    target = set(model.state_dict())
    candidates = [state]
    for prefix in ("model.", "model.model.", "module."):
        candidates.append({
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state.items()
        })
    return max(candidates, key=lambda item: len(target.intersection(item)))


def load_yolo(torch, checkpoint, thermal):
    # Import the custom class before torch.load so thermal checkpoints can be
    # reconstructed. This direct load does not invoke Predictor/Validator fuse.
    from common.yolo_support import ThermalDetectionModel  # noqa: F401
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model
    if hasattr(model, "is_fused") and model.is_fused():
        raise RuntimeError("YOLO checkpoint가 이미 fused 상태여서 원래 학습 모델 Params를 측정할 수 없습니다.")
    if thermal and not hasattr(model, "thermal_module"):
        raise RuntimeError("YOLO Full checkpoint에 thermal_module이 없습니다.")
    model = model.cuda().eval()
    image = torch.rand(1, 3, config["size"], config["size"], device="cuda")
    return model, (image,), {}


def load_rfdetr(torch, checkpoint, thermal):
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    from common.rfdetr_support import attach_rfdetr_thermal

    detector = RFDETRNano(pretrain_weights=None, num_classes=1, device="cpu")
    model = detector.model.model
    if thermal:
        attach_rfdetr_thermal(model, "full")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = align_state_dict(checkpoint_state(payload), model)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"RF-DETR checkpoint mismatch: {incompatible}")
    if thermal and (not hasattr(model, "thermal_module") or not hasattr(model, "thermal_lambda")):
        raise RuntimeError("RF-DETR Full의 thermal_module/thermal_lambda가 로드되지 않았습니다.")
    model = model.cuda().eval()
    image = torch.rand(1, 3, config["size"], config["size"], device="cuda")
    mean = image.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    nested = nested_tensor_from_tensor_list([(image[0] - mean) / std])
    return model, (nested,), {}


def load_rtdetr(torch, checkpoint, thermal):
    vendor = ROOT / "third_party" / "RT-DETR" / "rtdetr_pytorch"
    sys.path.insert(0, str(vendor))
    from src.core import YAMLConfig
    from common.rtdetr_support import attach_rtdetr_thermal, configure_rtdetr

    yaml_file = vendor / "configs" / "rtdetr" / "rtdetr_r34vd_6x_coco.yml"
    # Empty tuning prevents a pretrained download; only selected best.pt is used.
    cfg = YAMLConfig(str(yaml_file), tuning="")
    configure_rtdetr(
        cfg, ROOT / "rtdetr_baseline" / "generated",
        Path(os.environ["PROFILE_TEMP"]), "val"
    )
    model = cfg.model
    if thermal:
        attach_rtdetr_thermal(model, "full")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = align_state_dict(checkpoint_state(payload), model)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"RT-DETR checkpoint mismatch: {incompatible}")
    if thermal and (not hasattr(model, "thermal_module") or not hasattr(model, "thermal_lambda")):
        raise RuntimeError("RT-DETR Full의 thermal_module/thermal_lambda가 로드되지 않았습니다.")
    model = model.cuda().eval()
    image = torch.rand(1, 3, config["size"], config["size"], device="cuda")
    return model, (image,), {}


def verify_thermal_forward(model, family):
    thermal = getattr(model, "thermal_module", None)
    if thermal is None or not getattr(thermal, "last_debug", None):
        raise RuntimeError("Full 모델 forward에서 thermal_module 호출이 확인되지 않았습니다.")
    if family in ("rfdetr", "rtdetr"):
        if not hasattr(model, "thermal_lambda"):
            raise RuntimeError("DETR Full 모델에 thermal_lambda parameter가 없습니다.")
        attention_called = any(bool(getattr(module, "thermal_debug", None)) for module in model.modules())
        if not attention_called:
            raise RuntimeError("DETR Full forward에서 lambda attention bias 호출이 확인되지 않았습니다.")


class SupplementCounter:
    '''Shape-driven FLOPs for operations unsupported by torch.profiler.

    Convention: one floating-point add/multiply/divide/transcendental is one
    FLOP, hence one multiply-accumulate is two FLOPs. Bilinear grid sampling is
    four value multiplies plus three accumulation adds (7 FLOPs/output value).
    Coordinate/index/bounds calculations are not floating tensor arithmetic and
    are excluded, consistently with profiler FLOPs.
    '''

    def __init__(self, model, family, thermal):
        self.model = model
        self.family = family
        self.thermal = thermal
        self.parts = {"grid_sample_bilinear": 0, "deformable_weighted_aggregation": 0,
                      "thermal_equations": 0, "thermal_feature_enhancement": 0,
                      "thermal_attention_bias": 0}
        self.grid_calls = []
        self.handles = []
        self.original_grid_sample = None
        self.enhance_patches = []

    def _thermal_hook(self, module, inputs, output):
        image = inputs[0]
        pixels = image.shape[0] * image.shape[2] * image.shape[3]
        channels = image.shape[1]
        kernel = module.kernel_size * module.kernel_size
        # Eq.1 channel sum+division; Eq.2 pool sum+division; Eq.3 subtract+ReLU;
        # Eqs.4-5 two differences; Eqs.6-7 squares/add/epsilon/sqrt/neg/exp;
        # Eq.8 scalar products/sums and sigmoid (neg/exp/add/div).
        flops = pixels * ((channels - 1) + 1)
        flops += pixels * ((kernel - 1) + 1)
        flops += pixels * 2
        flops += pixels * 2
        flops += pixels * 5
        flops += pixels * 2
        flops += pixels
        if module.beta is not None:
            flops += pixels * 2
        if module.gamma is not None:
            flops += pixels * 2
        flops += pixels * 4
        # RF-DETR restores ImageNet-normalized RGB using multiply+add per value.
        if self.family == "rfdetr":
            flops += image.numel() * 2
        self.parts["thermal_equations"] += int(flops)

    def _deform_hook(self, module, inputs, output):
        query = inputs[0]
        batch, queries = int(query.shape[0]), int(query.shape[1])
        heads = int(getattr(module, "n_heads", getattr(module, "num_heads", 0)))
        levels = int(getattr(module, "n_levels", getattr(module, "num_levels", 0)))
        points = int(getattr(module, "n_points", getattr(module, "num_points", 0)))
        d_model = int(getattr(module, "d_model", heads * int(getattr(module, "head_dim", 0))))
        if not all((heads, levels, points, d_model)):
            return
        head_dim = d_model // heads
        positions = levels * points
        vectors = batch * queries * heads * head_dim
        # sampled_value * attention, then reduction across level*point.
        self.parts["deformable_weighted_aggregation"] += vectors * (positions + positions - 1)
        debug = getattr(module, "thermal_debug", None)
        if self.thermal and debug and "sampled_W" in debug:
            samples = debug["sampled_W"].numel()
            # [0,1] -> [-1,1]: mul+sub for two coordinates = 4/sample;
            # lambda*sampled_W + logits = 2/sample.
            self.parts["thermal_attention_bias"] += int(samples * 6)

    def _enhance_wrapper(self, original):
        def wrapped(feature, weight):
            result = original(feature, weight)
            aligned = feature.shape[0] * feature.shape[-2] * feature.shape[-1]
            # Bilinear W resize (7/output), 1+W, and feature*(1+W).
            self.parts["thermal_feature_enhancement"] += int(
                aligned * 7 + aligned + feature.numel()
            )
            return result
        return wrapped

    def __enter__(self):
        import torch.nn.functional as functional
        self.original_grid_sample = functional.grid_sample

        def grid_sample(input, grid, *args, **kwargs):
            output = self.original_grid_sample(input, grid, *args, **kwargs)
            mode = kwargs.get("mode", args[0] if args else "bilinear")
            self.grid_calls.append((tuple(input.shape), tuple(grid.shape), tuple(output.shape), mode))
            if mode == "bilinear":
                self.parts["grid_sample_bilinear"] += int(output.numel() * 7)
            return output

        functional.grid_sample = grid_sample
        for module in self.model.modules():
            if module.__class__.__name__ == "ThermalModule":
                self.handles.append(module.register_forward_hook(self._thermal_hook))
            if module.__class__.__name__ in ("MSDeformAttn", "MSDeformableAttention"):
                self.handles.append(module.register_forward_hook(self._deform_hook))
        for name in ("common.yolo_support", "common.rfdetr_support", "common.rtdetr_support"):
            loaded = sys.modules.get(name)
            if loaded is not None and hasattr(loaded, "enhance_feature"):
                original = loaded.enhance_feature
                self.enhance_patches.append((loaded, original))
                loaded.enhance_feature = self._enhance_wrapper(original)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        import torch.nn.functional as functional
        functional.grid_sample = self.original_grid_sample
        for loaded, original in self.enhance_patches:
            loaded.enhance_feature = original
        for handle in self.handles:
            handle.remove()

    @property
    def total(self):
        return sum(self.parts.values())


def measure():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    checkpoint = Path(config["checkpoint"])
    family = config["family"]
    loaders = {"yolo": load_yolo, "rfdetr": load_rfdetr, "rtdetr": load_rtdetr}
    model, args, kwargs = loaders[family](torch, checkpoint, config["thermal"])
    model.eval()

    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[{config['name']}] checkpoint load 완료, parameters={params:,}", flush=True)

    with torch.inference_mode():
        # First forward validates both the input contract and thermal execution.
        model(*args, **kwargs)
        if config["thermal"]:
            verify_thermal_forward(model, family)
        for _ in range(config["warmup"] - 1):
            model(*args, **kwargs)
    torch.cuda.synchronize()

    samples = []
    with torch.inference_mode():
        for _ in range(config["repeat"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(*args, **kwargs)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    supplement = SupplementCounter(model, family, config["thermal"])
    with supplement, torch.inference_mode(), torch.profiler.profile(
        activities=activities, record_shapes=True, with_flops=True
    ) as profiler:
        model(*args, **kwargs)
        torch.cuda.synchronize()

    events = profiler.key_averages(group_by_input_shape=True)
    total_flops = sum(int(getattr(event, "flops", 0) or 0) for event in events)
    custom_flops = supplement.total
    combined_flops = total_flops + custom_flops
    gflops = combined_flops / 1e9

    # PyTorch profiler reports FLOPs only for operators with a registered formula.
    # Deformable attention commonly includes grid-sample/custom kernels without
    # one, so DETR values are flagged as lower bounds rather than silently being
    # presented as complete counts.
    grid_events = [event for event in events if "grid_sampler" in event.key.lower()]
    if any((getattr(event, "flops", 0) or 0) > 0 for event in grid_events):
        # Future profiler versions may gain a formula. Never count it twice.
        custom_flops -= supplement.parts["grid_sample_bilinear"]
        supplement.parts["grid_sample_bilinear"] = 0
        combined_flops = total_flops + custom_flops
        gflops = combined_flops / 1e9

    audit_patterns = {
        "conv": ("aten::conv2d", "aten::convolution", "aten::_convolution"),
        "linear": ("aten::linear", "aten::addmm"),
        "matmul": ("aten::matmul", "aten::mm"),
        "bmm": ("aten::bmm",),
        "einsum": ("aten::einsum",),
        "grid_sample": ("aten::grid_sampler", "aten::grid_sampler_2d"),
    }
    operation_audit = {}
    for label, names in audit_patterns.items():
        matched = [event for event in events if event.key in names]
        operation_audit[label] = {
            "observed_ops": sorted({event.key for event in matched}),
            "profiler_flops": sum(int(getattr(event, "flops", 0) or 0) for event in matched),
            "observed_call_groups": sum(event.count for event in matched),
        }
    operation_audit["grid_sample"]["custom_flops"] = supplement.parts["grid_sample_bilinear"]
    operation_audit["deformable_weighted_aggregation"] = {
        "custom_flops": supplement.parts["deformable_weighted_aggregation"]
    }

    floating_without_formula = {
        "_batch_norm_impl_index", "_softmax", "add", "avg_pool2d", "batch_norm",
        "cudnn_batch_norm", "cumsum", "div", "exp", "floor_divide", "gelu",
        "layer_norm", "mean", "mul", "native_layer_norm", "pow", "relu",
        "sigmoid", "silu", "softmax", "sqrt", "sub", "sum",
        "upsample_bilinear2d",
    }
    handled_grid = supplement.parts["grid_sample_bilinear"] > 0
    uncovered = sorted({
        event.key for event in events
        if not (getattr(event, "flops", 0) or 0)
        and event.key.startswith("aten::")
        and event.key.removeprefix("aten::").rstrip("_") in floating_without_formula
        and not (handled_grid and "grid_sampler" in event.key.lower())
    })
    warnings = []
    if gflops <= 0.0 or gflops < 1.0:
        warnings.append(f"비정상적으로 작은 profiler FLOPs 값({gflops:.6f} GFLOPs)")
    if uncovered:
        warnings.append(
            "custom counter로도 정확히 분리할 수 없는 profiler 미지원 연산이 남아 "
            "GFLOPs exact=False입니다: " + ", ".join(uncovered[:12])
        )
    for warning in warnings:
        print(f"[GFLOPs WARNING] {config['name']}: {warning}", flush=True)

    index = torch.cuda.current_device()
    return {
        "model": config["name"],
        "params": params,
        "params_m": params / 1e6,
        "thermal_parameter_names": [
            name for name, _ in model.named_parameters()
            if name == "thermal_lambda" or name.startswith("thermal_module.")
        ],
        "profiler_flops": total_flops,
        "custom_added_flops": custom_flops,
        "custom_flops_breakdown": supplement.parts,
        "total_flops": combined_flops,
        "gflops": gflops,
        "gflops_exact": not uncovered,
        "unsupported_ops": uncovered,
        "grid_sample_calls": supplement.grid_calls,
        "operation_audit": operation_audit,
        "flops_convention": "add=1, multiply=1, MAC=2 FLOPs; bilinear sample=7 FLOPs/output value",
        "latency_ms": sum(samples) / len(samples),
        "gflops_warning": " | ".join(warnings),
        "batch_size": 1,
        "warmup": config["warmup"],
        "repeat": config["repeat"],
        "input_resolution": config["size"],
        "checkpoint": str(checkpoint),
        "gpu": torch.cuda.get_device_name(index),
    }


try:
    result = measure()
    RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
except BaseException as error:
    traceback.print_exc()
    RESULT_FILE.write_text(json.dumps({
        "model": config.get("name", "unknown"),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    raise
"""


def checkpoint_for(run: dict) -> Path:
    path = ROOT / run["checkpoint"].format(seed=PROFILE_SEED)
    if not path.is_file():
        raise FileNotFoundError(
            f"{run['name']}: PROFILE_SEED={PROFILE_SEED} checkpoint가 없습니다.\n예상 경로: {path}"
        )
    return path.resolve()


def run_one(run: dict, python: Path, worker: Path, temp_dir: Path) -> dict:
    checkpoint = checkpoint_for(run)
    safe_name = run["name"].lower().replace(" ", "_").replace("-", "_")
    config_file = temp_dir / f"{safe_name}.config.json"
    result_file = temp_dir / f"{safe_name}.result.json"
    config = {**run, "checkpoint": str(checkpoint), "warmup": WARMUP, "repeat": REPEAT}
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["PROFILE_ROOT"] = str(ROOT)
    env["PROFILE_TEMP"] = str(temp_dir)
    # Avoid importing a caller-specific sitecustomize/PYTHONPATH into model env.
    env.pop("PYTHONPATH", None)

    print("\n" + "=" * 88)
    print(f"[RUN] {run['name']}")
    print(f"Environment : {run['env']}")
    print(f"Python      : {python}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Input       : 1x3x{run['size']}x{run['size']}")
    print("=" * 88, flush=True)

    completed = subprocess.run(
        [str(python), str(worker), str(config_file), str(result_file)],
        cwd=ROOT, env=env
    )
    if not result_file.is_file():
        raise RuntimeError(f"worker 결과가 없습니다 (return code={completed.returncode})")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if completed.returncode != 0 or "error" in result:
        raise RuntimeError(
            f"{run['name']} profiling 실패 (return code={completed.returncode})\n"
            f"{result.get('error', 'unknown error')}\n{result.get('traceback', '')}"
        )
    return result


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 116)
    print("FINAL PROFILE RESULTS")
    print("=" * 116)
    print(
        f"{'Model':<30}{'Params':>16}{'Params (M)':>16}"
        f"{'GFLOPs':>14}{'Latency (ms)':>17}{'GFLOPs Exact':>17}"
    )
    print("-" * 116)
    for result in results:
        if "error" in result:
            print(f"{result['model']:<30}{'ERROR':>16}{'ERROR':>16}{'ERROR':>14}{'ERROR':>17}{'ERROR':>17}")
        else:
            print(
                f"{result['model']:<30}{result['params']:>16,}{result['params_m']:>16.6f}"
                f"{result['gflops']:>14.6f}{result['latency_ms']:>17.4f}"
                f"{str(result['gflops_exact']):>17}"
            )
    print("=" * 116)
    for result in results:
        if result.get("gflops_warning"):
            print(f"WARNING - {result['model']}: {result['gflops_warning']}")
        if result.get("error"):
            print(f"ERROR   - {result['model']}: {result['error']}")


def save_csv(results: list[dict]) -> Path:
    output = ROOT / "profile_results.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Model", "Params", "Params (M)", "GFLOPs", "Latency (ms)",
            "GFLOPs Exact", "Unsupported Ops",
        ])
        for result in results:
            if "error" in result:
                writer.writerow([result["model"], "ERROR", "ERROR", "ERROR", "ERROR", "False", result["error"]])
            else:
                writer.writerow([
                    result["model"], str(result["params"]), f"{result['params_m']:.6f}",
                    f"{result['gflops']:.6f}", f"{result['latency_ms']:.4f}",
                    str(result["gflops_exact"]), "; ".join(result["unsupported_ops"]),
                ])
    return output


def save_details(results: list[dict]) -> Path:
    output = ROOT / "profile_details.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def validate_parameter_deltas(results: list[dict]) -> list[str]:
    warnings = []
    expected = ((0, 1, 3), (2, 3, 3), (4, 5, 4), (6, 7, 4))
    for baseline_index, full_index, expected_delta in expected:
        baseline, full = results[baseline_index], results[full_index]
        if "error" in baseline or "error" in full:
            continue
        delta = full["params"] - baseline["params"]
        if delta != expected_delta:
            names = full.get("thermal_parameter_names", [])
            warning = (
                f"PARAM WARNING - {full['model']}: 실제 Params 차이={delta:+,}, "
                f"예상={expected_delta:+,}, thermal parameters={names}"
            )
            warnings.append(warning)
            print(warning)
        else:
            print(
                f"PARAM CHECK   - {full['model']}: Baseline 대비 {delta:+,} "
                f"({', '.join(full.get('thermal_parameter_names', []))})"
            )
        flop_delta = full["total_flops"] - baseline["total_flops"]
        if flop_delta <= 0:
            warning = (
                f"GFLOPs WARNING - {full['model']}: Baseline 대비 FLOPs 증가량이 "
                f"{flop_delta / 1e9:+.9f} GFLOPs입니다."
            )
            warnings.append(warning)
            print(warning)
        else:
            print(f"GFLOPs CHECK  - {full['model']}: Baseline 대비 {flop_delta / 1e9:+.6f} GFLOPs")
    return warnings


def main() -> None:
    print(
        "MODEL PROFILING\n"
        f"seed={PROFILE_SEED}, batch=1, warm-up={WARMUP}, repeat={REPEAT}\n"
        "latency=model forward only, GFLOPs=PyTorch profiler single forward"
    )

    # Resolve all environments and exact checkpoints before loading any model.
    pythons = {name: get_env_python(name) for name in (ENV_YOLO, ENV_RFDETR, ENV_RTDETR)}
    for run in RUNS:
        checkpoint_for(run)

    results = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="thermal_profile_") as value:
        temp_dir = Path(value)
        worker = temp_dir / "profile_worker.py"
        worker.write_text(WORKER_CODE, encoding="utf-8")
        for run in RUNS:
            try:
                results.append(run_one(run, pythons[run["env"]], worker, temp_dir))
            except Exception as error:
                traceback.print_exc()
                failures.append(run["name"])
                results.append({"model": run["name"], "error": str(error)})

    parameter_warnings = validate_parameter_deltas(results)
    print_table(results)
    csv_path = save_csv(results)
    details_path = save_details(results)
    print(f"\nCSV 저장 완료: {csv_path}")
    print(f"상세 JSON 저장 완료: {details_path}")
    if parameter_warnings:
        print("\n".join(parameter_warnings))
    if failures:
        raise RuntimeError("profiling 실패 모델: " + ", ".join(failures))


if __name__ == "__main__":
    main()
