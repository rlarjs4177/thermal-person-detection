# YOLO11m + thermal module

논문 식 (1)~(8)로 만든 `W`를 공식 YOLO11 `Detect` 입력 feature map에 bilinear resize하여 식 (9)를 적용합니다. 숫자 layer index가 아니라 공식 `Detect` 타입을 찾아 삽입하므로 임의 neck 재구현이 아닙니다.

```powershell
pip install -r requirements.txt
python train.py --thermal-mode heatcore --seed 0
python train.py --thermal-mode heatcore_diffusion --seed 0
python train.py --thermal-mode full --seed 0
python evaluate.py --checkpoint outputs\full\seed_0\checkpoints\best.pt --split test
python profile.py --thermal-mode full
```

full은 baseline보다 `alpha,beta,gamma` 3개가 추가되어 총 20,053,782 parameters입니다. 모드별 결과는 `outputs/<mode>/seed_<N>/`에 완전히 분리됩니다. 학습된 scalar는 매 epoch `metrics.csv`와 최종 `results.json`에 저장됩니다. 전체 수식과 공통 조건은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `ultralytics/ultralytics`, PyPI `ultralytics==8.4.129`, release commit `509fac1`
- variant/checkpoint: YOLO11m / `yolo11m.pt`; SHA256 `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 640×640
- early stopping: val mAP50-95, max, min_delta 0, patience 100
- checkpoint: `outputs/<mode>/seed_<N>/checkpoints/{best.pt,last.pt,epoch_0100.pt,...}`
- scalar 확인: `metrics.csv`/`results.json`; 비활성 항과 lambda는 `null`; profile은 `python profile.py --thermal-mode <mode>`
