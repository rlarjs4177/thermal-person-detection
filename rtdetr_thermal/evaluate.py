import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common.rtdetr_cli import evaluate_rtdetr
if __name__=="__main__": evaluate_rtdetr(__file__,True)
