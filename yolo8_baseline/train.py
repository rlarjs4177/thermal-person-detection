import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.yolo_cli import train_yolo
if __name__ == "__main__": train_yolo(__file__, "yolov8m.pt", thermal=False)
