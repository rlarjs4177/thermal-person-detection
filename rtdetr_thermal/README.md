# original RT-DETR-R34 + thermal module

원 저자 PResNet backbone 출력과 HybridEncoder 사이에 식 (9)를 적용합니다. decoder의 공식 `MSDeformableAttention` sampling location으로 `W(p)`를 미분 가능하게 sampling하고 attention softmax 직전 식 (10)을 더합니다. thermal/detector 공간 정렬을 위해 multi-scale training은 비활성화했습니다.

```powershell
pip install -r requirements.txt
python train.py --thermal-mode heatcore --seed 0
python train.py --thermal-mode heatcore_diffusion --seed 0
python train.py --thermal-mode full --seed 0
python evaluate.py --checkpoint outputs\full\seed_0\checkpoints\best.pt --split test --thermal-mode full
python profile.py --thermal-mode full
```

full은 `alpha,beta,gamma,lambda` 네 scalar를 추가해 총 31,319,933 parameters입니다. `metrics.csv`에는 validation COCO AP, precision/recall, LR, scalar가 기록되고 `checkpoints/`에는 best/last/100-epoch 주기 checkpoint가 생성됩니다. 결과는 모드·seed별로 분리됩니다. 수식·padding·전처리 근거는 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `lyuwenyu/RT-DETR` original PyTorch, vendored commit `068dfde65f2667ad6555883c69d73de886518cad` (RT-DETRv2 아님)
- variant/checkpoint: RT-DETR-R34 / `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth`; SHA256 `FB167E502CE47AD6959BFD691225C0DDF318F7C346AE419DC1CA4C218764B578`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 640×640
- early stopping: val mAP50-95, max, min_delta 0, patience 100
- checkpoint: `outputs/<mode>/seed_<N>/checkpoints/{best.pt,last.pt,epoch_0100.pt,...}`; full training state와 scalar 포함
- scalar 확인: `metrics.csv`/`results.json`; 비활성 항은 `null`, lambda는 모든 transformer thermal mode에서 학습·기록; profile은 위 명령
