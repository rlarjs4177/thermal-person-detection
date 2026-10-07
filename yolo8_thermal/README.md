# YOLOv8m + thermal module

YOLOv8m의 공식 `Detect` 입력 feature map에 논문 식 (9)를 적용합니다. 식 (1)~(8)의 `W`는 증강 후 `[0,1]` 입력에서 계산되므로 검출 branch와 공간적으로 정렬됩니다. YOLO는 deformable cross-attention이 없어 식 (10)을 적용하지 않습니다.

```powershell
pip install -r requirements.txt
python train.py --thermal-mode heatcore --seed 0
python train.py --thermal-mode heatcore_diffusion --seed 0
python train.py --thermal-mode full --seed 0
python evaluate.py --checkpoint outputs\full\seed_0\checkpoints\best.pt --split test
python profile.py --thermal-mode full
```

모드별 경로는 `outputs/<thermal-mode>/seed_<N>/`입니다. `metrics.csv`와 `results.json`에서 학습된 `alpha`, `beta`, `gamma`를 확인할 수 있습니다. full 모델은 baseline보다 정확히 3개의 학습 scalar가 추가되어 총 25,856,902 parameters입니다. 구현 위치는 `thermal_module.py`, `common/thermal.py`, `common/yolo_support.py`이며 전체 수식·padding·학습 조건은 [상위 README](../README.md)에 설명했습니다.

## 재현 정보

- 공식 source: `ultralytics/ultralytics`, PyPI `ultralytics==8.4.129`, release commit `509fac1`
- variant/checkpoint: YOLOv8m / `yolov8m.pt`; SHA256 `5D4A90CDC7A21786CC59CD19778E9EAFFF836DF9E2DA32524737C7EE6EFE4FE5`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 640×640
- early stopping: val mAP50-95 최대화, min_delta 0, patience 100
- checkpoint: `outputs/<mode>/seed_<N>/checkpoints/{best.pt,last.pt,epoch_0100.pt,...}`
- scalar 확인: 같은 run의 `metrics.csv`와 `results.json`; 비활성 beta/gamma 및 YOLO의 lambda는 `null`
- 평가/profile: 위 `evaluate.py` 명령 / `python profile.py --thermal-mode <mode>`
