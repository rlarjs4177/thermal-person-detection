# RF-DETR Nano + thermal module

공식 RF-DETR backbone projector 출력과 transformer 사이에 식 (9)를 적용하고, 공식 `MSDeformAttn`의 sampling location에서 `W(p)`를 bilinear sampling해 softmax 직전 logit에 식 (10)을 적용합니다. ImageNet normalization을 정확히 역변환하여 공간 증강 후 `[0,1]` thermal branch를 복원합니다.

```powershell
pip install -r requirements.txt
python train.py --thermal-mode heatcore --seed 0
python train.py --thermal-mode heatcore_diffusion --seed 0
python train.py --thermal-mode full --seed 0
python evaluate.py --checkpoint outputs\full\seed_0\checkpoints\best.pt --thermal-mode full
python profile.py --thermal-mode full
```

full은 baseline에 `alpha,beta,gamma,lambda` 네 scalar를 추가해 총 30,467,302 parameters입니다. `profile.py`는 sampling location, attention logits, sampled `W` shape와 `bias_before_softmax=True`를 출력합니다. 결과는 `outputs/<mode>/seed_<N>/`에 분리되며 학습 scalar는 `metrics.csv`와 `results.json`에 기록됩니다. 자세한 수식과 조건은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `roboflow/rf-detr`, PyPI `rfdetr==1.9.0`, release commit `a700efc`
- variant/checkpoint: RF-DETR Nano / `rf-detr-nano.pth`; SHA256 `D8D6B9EE57D4D0ED2B1F305163624712A0532CB7BCE0C747317984FC5457440D`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 384×384
- early stopping: val mAP50-95, max, min_delta 0, patience 100
- checkpoint: `outputs/<mode>/seed_<N>/checkpoints/{best.pt,last.pt,epoch_0100.pt,...}`; model/optimizer/scheduler/scaler와 scalar 포함
- scalar 확인: `metrics.csv`/`results.json`; heatcore의 beta/gamma, heatcore_diffusion의 gamma는 `null`; profile은 위 명령
