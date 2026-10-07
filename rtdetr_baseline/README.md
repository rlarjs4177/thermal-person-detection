# original RT-DETR-R34 baseline

공식 Ultralytics RT-DETR이 아니라 원 저자 `lyuwenyu/RT-DETR`의 PyTorch R34 구성과 공식 `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth` checkpoint를 사용합니다. vendored source는 `third_party/RT-DETR`에 고정했습니다.

```powershell
pip install -r requirements.txt
python train.py --seed 0
python evaluate.py --checkpoint outputs\seed_0\checkpoints\best.pt --split test
python profile.py
```

입력 640×640, person 1 class, 1,000 epochs, batch 16, AdamW, LR `1e-4`, weight decay `1e-4`, patience 100입니다. COCO JSON만 `generated/`에 만들며 원본은 변경하지 않습니다. 검증된 1-class 모델 파라미터 수는 31,319,929개입니다. 출력 규격과 공식 checkpoint 전이 시 head mismatch 설명은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `lyuwenyu/RT-DETR` original PyTorch, vendored commit `068dfde65f2667ad6555883c69d73de886518cad` (RT-DETRv2 아님)
- variant/checkpoint: RT-DETR-R34 / `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth`
- checkpoint SHA256: `FB167E502CE47AD6959BFD691225C0DDF318F7C346AE419DC1CA4C218764B578`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 640×640
- checkpoint: val mAP50-95 기준 `best.pt`, `last.pt`, 매 100 epoch `epoch_XXXX.pt`; optimizer/scheduler/scaler/seed/config 포함
- baseline scalar는 모두 `null`; profile은 `python profile.py`
