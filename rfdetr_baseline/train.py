import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.rfdetr_cli import train_rfdetr
if __name__ == "__main__": train_rfdetr(__file__, thermal=False)
