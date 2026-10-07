# Thermal Person Detection

### 열화상 사람 탐지를 위한 YOLO · DETR 기반 비교 실험

열화상 이미지에서 사람을 탐지하는 모델에 **국소 밝기 대비와 공간 변화 정보를 활용하는 열 특성 모듈**을 결합한 연구 프로젝트입니다.

YOLOv8m, YOLO11m, RF-DETR Nano, RT-DETR-R34의 네 모델을 대상으로 기본 모델(Baseline)과 열 특성 모듈을 추가한 모델(Thermal)을 구성했습니다. 총 8개 실험 구성을 통해 모델 구조에 따른 적용 방식과 열 특성의 기여를 비교할 수 있도록 정리했습니다.

## 🔍 프로젝트 배경

열화상 사람 탐지에서는 사람과 배경 사이의 밝기 차이, 신체 주변의 경계, 주변 영역과의 상대적인 대비가 중요한 단서가 될 수 있습니다. 이 프로젝트는 이러한 영상 특성을 명시적인 가중치 맵으로 만들고, 기존 검출 모델의 특징과 결합하는 방법을 살펴봅니다.

주요 비교 관점은 다음과 같습니다.

- 같은 검출 모델에서 열 특성 모듈의 유무가 탐지 결과에 미치는 영향
- YOLO 계열과 Transformer 기반 DETR 계열의 서로 다른 결합 방식
- 국소 대비, 공간 변화량, 변화량 기반 감쇠 항의 단계별 기여
- 탐지 지표와 함께 확인하는 모델 파라미터 및 학습된 열 특성 계수

## 🧩 비교 모델

각 모델은 Baseline과 Thermal 구성을 별도로 갖습니다. Thermal 구성 안에서는 세 가지 모드로 구성 요소를 나누어 비교합니다.

| 모델 | 기반 구현 | 입력 크기 | Baseline | Thermal 결합 위치 |
|---|---|---|---|---|
| YOLOv8m | Ultralytics | 640 × 640 | 열 모듈 없음 | Detect 헤드에 전달되는 특징 맵 |
| YOLO11m | Ultralytics | 640 × 640 | 열 모듈 없음 | Detect 헤드에 전달되는 특징 맵 |
| RF-DETR Nano | Roboflow RF-DETR | 384 × 384 | 열 모듈 없음 | Backbone projector 출력과 deformable attention |
| RT-DETR-R34 | 원 저자 RT-DETR PyTorch | 640 × 640 | 열 모듈 없음 | PResNet backbone 출력과 deformable attention |

Baseline과 Thermal의 비교를 위해 학습 코드에는 최대 1,000 epochs, batch size 16, AdamW, 초기 학습률 `1e-4`, weight decay `1e-4`, validation mAP50–95 기반 early stopping을 설정했습니다. 입력 크기와 세부 학습 처리는 각 모델의 구현에 따라 달라집니다.

## 🌡️ 열 특성 모듈

공통 모듈은 RGB로 표현된 열화상 입력의 채널 평균을 구한 뒤, 세 가지 영상 특성을 계산합니다.

1. **국소 대비(Heat Core, H)**  
   각 픽셀과 7 × 7 주변 평균의 차이 중 양수 부분을 사용합니다. 주변보다 밝은 영역을 나타냅니다.
2. **공간 변화량(Diffusion, D)**  
   가로·세로 방향의 인접 픽셀 차이로 gradient magnitude를 계산합니다. 코드에서는 `diffusion`이라는 이름을 사용합니다.
3. **감쇠 항(Decay, R)**  
   공간 변화량에 `exp(-D)`를 적용한 항입니다.

학습 가능한 계수 α, β, γ가 이 특성들을 결합하고, sigmoid를 통해 공간 가중치 맵 W를 만듭니다. 이 특성들은 영상 밝기에서 계산되며, 물리적 온도나 실제 열 확산량을 직접 측정하는 값은 아닙니다.

### 특징 맵과 Attention에 결합

- **공통 특징 강화:** W를 특징 맵 크기에 맞게 보간한 뒤 `F × (1 + W)` 형태로 기존 특징을 조절합니다.
- **DETR 계열 Attention 보정:** deformable attention의 sampling location에서 W를 추출하고, 학습 가능한 λ를 곱해 softmax 이전 attention logit에 더합니다.
- **YOLO 계열:** Detect 헤드 입력 특징을 강화하며, DETR용 attention 보정은 사용하지 않습니다.

### Ablation 구성

| 모드 | 사용 특성 | 가중치 맵의 학습 계수 |
|---|---|---|
| `heatcore` | 국소 대비 H | α |
| `heatcore_diffusion` | 국소 대비 H + 공간 변화량 D | α, β |
| `full` | 국소 대비 H + 공간 변화량 D + 감쇠 항 R | α, β, γ |

사용하지 않는 항의 계수는 생성하지 않습니다. DETR의 Thermal 구성에는 이 계수들과 별도로 attention 보정 계수 λ가 추가됩니다.

## 📊 평가와 기록

코드는 학습·검증·평가 과정을 구분하고, 다음 정보를 기록하도록 구성되어 있습니다.

- mAP50, mAP50–95, Precision, Recall
- 학습 손실, 학습률, epoch 및 best checkpoint
- 열 특성 계수 α, β, γ와 DETR attention 계수 λ
- 모델 파라미터 수 및 프로파일링 정보

실험 결과는 모델, Thermal 모드, seed별로 구분합니다. Precision과 Recall의 집계 방식은 모델별 평가 구현에 따라 다를 수 있으므로, 모델 간 비교에서는 지표 정의와 평가 조건을 함께 확인해야 합니다.

이 저장소의 개요에는 수치 성능표를 포함하지 않습니다. 모듈의 성능 개선 여부나 일반화 효과는 동일한 데이터와 평가 조건에서 별도로 검증해야 합니다.

## 🗂️ 코드 구성

- **`common`**: 데이터 처리, 열 특성 계산, 모델별 학습·평가 adapter와 공통 기록 기능
- **`yolo8_baseline` / `yolo8_thermal`**: YOLOv8m 비교 구성
- **`yolo11_baseline` / `yolo11_thermal`**: YOLO11m 비교 구성
- **`rfdetr_baseline` / `rfdetr_thermal`**: RF-DETR Nano 비교 구성
- **`rtdetr_baseline` / `rtdetr_thermal`**: RT-DETR-R34 비교 구성
- **`third_party/RT-DETR`**: 원 저자 RT-DETR 기반 코드

각 모델 폴더의 진입점은 공통 adapter를 호출합니다. 공유 로직을 `common`으로 모으고, 모델별 차이는 adapter와 설정으로 구분한 구조입니다.

## 🖼️ 데이터와 공개 범위

사람(Person) 단일 클래스의 열화상 JPG 이미지와 JSON bounding-box annotation을 대상으로 합니다. Annotation은 좌상단 기준의 절대 좌표 `(x, y, width, height)`를 사용하며, 학습·검증·테스트 split을 구분합니다. YOLO는 원본 JSON을 읽고, DETR 계열은 해당 annotation을 COCO 형식의 메타데이터로 변환합니다.

공개 저장소에는 소스 코드, 설정, 설명 문서를 포함합니다. 데이터셋, 원본 이미지, 학습 가중치, checkpoint, 실험 출력물과 개인 실험 기록은 배포하지 않습니다. 데이터와 가중치는 각 사용 조건에 따라 별도로 준비해야 합니다.

## 참고 구현 및 라이선스

- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLOv8m 및 YOLO11m
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr): RF-DETR Nano
- [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR): RT-DETR-R34 기반 구현

포함된 RT-DETR 코드의 Apache-2.0 라이선스는 [원본 LICENSE](third_party/RT-DETR/LICENSE)에 보존되어 있습니다. 제3자 코드에 관한 안내는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하십시오. 사용되는 외부 코드와 패키지에는 각 프로젝트의 라이선스가 적용됩니다. 이 저장소의 자체 작성 코드에는 별도의 라이선스를 지정하지 않았습니다.
