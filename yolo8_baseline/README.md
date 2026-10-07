# YOLOv8m baseline

Ultralytics `yolov8m.pt` COCO pretrained checkpoint를 person 1-class head로 전이학습하는 기준 모델입니다. 입력은 640×640이고 열 모듈은 없습니다.

```powershell
pip install -r requirements.txt
python train.py --seed 0
python evaluate.py --checkpoint outputs\seed_0\checkpoints\best.pt --split test
python profile.py
```

학습 조건은 1,000 epochs, batch 16, AdamW, LR `1e-4`, weight decay `1e-4`, patience 100입니다. 색/밝기 증강은 끄고 horizontal flip만 사용합니다. JSON bbox는 `common.yolo_support.JsonPersonDataset`이 원본에서 직접 읽어 정규화된 `xywh`로 변환합니다.

출력은 `outputs/seed_<N>/` 아래 `checkpoints/{best,last,epoch_XXXX}.pt`, `metrics.csv`, `results.json`, `config.json`입니다. 검증된 1-class 파라미터 수는 총 25,856,899개입니다. 공통 구현·데이터·주의사항은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `ultralytics/ultralytics`, PyPI `ultralytics==8.4.129`, release commit `509fac1`
- variant/checkpoint: YOLOv8m / `yolov8m.pt`
- checkpoint SHA256: `5D4A90CDC7A21786CC59CD19778E9EAFFF836DF9E2DA32524737C7EE6EFE4FE5`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class
- input: 640×640
- checkpoint: 최고 val mAP50-95는 `best.pt`, 매 epoch 최신은 `last.pt`, 도달한 100 epoch마다 `epoch_0100.pt` 형식
- baseline의 `alpha/beta/gamma/lambda`는 `results.json`에서 `null`; profile은 `python profile.py`
