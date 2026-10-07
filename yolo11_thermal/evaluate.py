import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.yolo_cli import evaluate_yolo
if __name__ == "__main__": evaluate_yolo(__file__, thermal=True)
