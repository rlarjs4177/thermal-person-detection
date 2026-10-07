# YOLO11m baseline

Ultralytics `yolo11m.pt` COCO pretrained checkpoint를 person 1-class head로 전이학습하는 기준 모델입니다. 입력은 640×640입니다.

```powershell
pip install -r requirements.txt
python train.py --seed 0
python evaluate.py --checkpoint outputs\seed_0\checkpoints\best.pt --split test
python profile.py
```

1,000 epochs, batch 16, AdamW, LR `1e-4`, weight decay `1e-4`, patience 100을 사용하며 color augmentation은 끄고 horizontal flip만 유지합니다. 출력은 `outputs/seed_<N>/`의 checkpoints, `metrics.csv`, `results.json`, `config.json`입니다. 검증된 1-class 파라미터 수는 20,053,779개입니다. 자세한 공통 조건은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `ultralytics/ultralytics`, PyPI `ultralytics==8.4.129`, release commit `509fac1`
- variant/checkpoint: YOLO11m / `yolo11m.pt`
- checkpoint SHA256: `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 640×640
- checkpoint: val mAP50-95 기준 `best.pt`, `last.pt`, 도달한 매 100 epoch의 `epoch_XXXX.pt`
- baseline thermal scalar는 `results.json`에서 모두 `null`; profile은 `python profile.py`
