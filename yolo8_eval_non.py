from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# 설정
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "non"
OUTPUT_DIR = INPUT_DIR / "results"

WEIGHTS = (
    ROOT
    / "yolo8_thermal"
    / "outputs"
    / "full"
    / "seed_0"
    / "weights"
    / "best.pt"
)

IMGSZ = 640
CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# Windows Arial 폰트
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_SIZE = 14


# --------------------------------------------------
# Arial 폰트 불러오기
# --------------------------------------------------
def load_font():
    if FONT_PATH.exists():
        return ImageFont.truetype(
            str(FONT_PATH),
            FONT_SIZE
        )

    # Arial을 찾지 못한 경우 기본 폰트 사용
    print(
        f"[WARNING] Arial font not found: {FONT_PATH}"
    )

    return ImageFont.load_default()


FONT = load_font()


# --------------------------------------------------
# Bounding Box 및 Label 시각화
# --------------------------------------------------
def draw_detection(image, xyxy, confidence):
    x1, y1, x2, y2 = map(int, xyxy)

    # Fig. 7 형식
    label = f"person:{confidence:.2f}"

    # OpenCV는 BGR
    green_bgr = (0, 255, 0)

    # --------------------------------------------------
    # Bounding Box
    # --------------------------------------------------
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        green_bgr,
        2
    )

    # --------------------------------------------------
    # OpenCV BGR -> PIL RGB
    # --------------------------------------------------
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    # 글자 크기
    bbox = draw.textbbox(
        (0, 0),
        label,
        font=FONT
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 기본적으로 bbox 위쪽에 표시
    text_x = x1
    text_y = y1 - text_height - 4

    # 위쪽 공간이 부족하면 bbox 안쪽
    if text_y < 0:
        text_y = y1 + 3

    # 오른쪽으로 글자가 이미지 밖으로 나가지 않도록 조정
    if text_x + text_width > image.shape[1]:
        text_x = max(
            0,
            image.shape[1] - text_width - 2
        )

    # --------------------------------------------------
    # Arial 글자 출력
    # 배경 박스 없이 초록색 글자만 표시
    # --------------------------------------------------
    draw.text(
        (text_x, text_y),
        label,
        font=FONT,
        fill=(0, 255, 0),
        stroke_width=0
    )

    # --------------------------------------------------
    # PIL RGB -> OpenCV BGR
    # --------------------------------------------------
    result = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    image[:] = result


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"non 폴더를 찾을 수 없습니다: {INPUT_DIR}"
        )

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLOv8 Thermal Full best.pt를 찾을 수 없습니다: {WEIGHTS}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # non 폴더 바로 아래의 이미지만 검색
    # results 폴더 내부 이미지는 포함되지 않음
    image_paths = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"non 폴더에 이미지가 없습니다: {INPUT_DIR}"
        )

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print("----------------------------------------")
    print("YOLOv8 Thermal Full - Non-human Test")
    print("----------------------------------------")
    print(f"Model  : {WEIGHTS}")
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Images : {len(image_paths)}")
    print(f"Device : {device}")
    print("----------------------------------------")
    print()

    # --------------------------------------------------
    # YOLOv8 Thermal Full checkpoint
    # --------------------------------------------------
    model = YOLO(
        str(WEIGHTS)
    )

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[SKIP] 이미지 로드 실패: "
                f"{image_path.name}"
            )
            continue

        # --------------------------------------------------
        # 추론
        # --------------------------------------------------
        result = model.predict(
            source=image,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False,
        )[0]

        output_image = image.copy()

        person_count = 0
        confidences = []

        # --------------------------------------------------
        # Detection 결과
        # --------------------------------------------------
        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls.item()
                )

                # Person = class 0
                if class_id != 0:
                    continue

                confidence = float(
                    box.conf.item()
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                draw_detection(
                    output_image,
                    xyxy,
                    confidence
                )

                person_count += 1

                confidences.append(
                    confidence
                )

        # --------------------------------------------------
        # 결과 저장
        # --------------------------------------------------
        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output_image
        )

        # --------------------------------------------------
        # Console 출력
        # --------------------------------------------------
        if person_count > 0:

            conf_text = ", ".join(
                f"{conf:.2f}"
                for conf in confidences
            )

            print(
                f"[SAVE] {image_path.name} "
                f"| person: {person_count} "
                f"| confidence: {conf_text}"
            )

        else:

            print(
                f"[SAVE] {image_path.name} "
                f"| person: 0"
            )

    print()
    print("----------------------------------------")
    print("Inference Complete")
    print(f"Results: {OUTPUT_DIR}")
    print("----------------------------------------")


if __name__ == "__main__":
    # numpy는 PIL/OpenCV 변환에 사용
    import numpy as np

    main()