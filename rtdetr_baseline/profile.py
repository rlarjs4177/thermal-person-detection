import sys; from pathlib import Path; _d=Path(__file__).resolve().parent; sys.path=[p for p in sys.path if Path(p or '.').resolve()!=_d]; sys.path.insert(0,str(_d.parent))
from common.rtdetr_cli import profile_rtdetr
if __name__=="__main__": profile_rtdetr(__file__,False)
