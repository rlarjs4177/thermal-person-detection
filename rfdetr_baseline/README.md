# RF-DETR Nano baseline

Roboflow 공식 RF-DETR 1.9.0의 Nano pretrained checkpoint를 사용합니다. 입력은 공식 Nano 해상도 384×384이며 학습용 head는 person 1 class입니다.

```powershell
pip install -r requirements.txt
python train.py --seed 0
python evaluate.py --checkpoint outputs\seed_0\checkpoints\best.pt
python profile.py
```

`generated/`에는 원본 이미지 절대 경로를 가리키는 COCO JSON만 생성됩니다. 원본 데이터는 수정하지 않습니다. 1,000 epochs, batch 16, AdamW, LR/encoder LR `1e-4`, weight decay `1e-4`, patience 100, horizontal flip만 사용합니다. 출력은 `outputs/seed_<N>/`에 저장됩니다. 공식 pretrained 구조의 검증된 파라미터 수는 30,467,298개입니다. 자세한 내용은 [상위 README](../README.md)를 보십시오.

## 재현 정보

- 공식 source: `roboflow/rf-detr`, PyPI `rfdetr==1.9.0`, release commit `a700efc`
- variant/checkpoint: RF-DETR Nano / `rf-detr-nano.pth`
- checkpoint SHA256: `D8D6B9EE57D4D0ED2B1F305163624712A0532CB7BCE0C747317984FC5457440D`
- dataset: `../data`; train 5,200 / val 2,000 / test 276, person 1 class; input 384×384
- checkpoint: val mAP50-95 기준 `best.pt`, `last.pt`, 매 100 epoch `epoch_XXXX.pt`; official training state를 포함해 resume 가능
- scalar는 모두 `null`; profile은 `python profile.py`
