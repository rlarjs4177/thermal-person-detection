import json
from pathlib import Path
from common.data import inspect_dataset, validate_white_hot

if __name__ == "__main__":
    root=Path(__file__).resolve().parent
    print(json.dumps({"dataset":inspect_dataset(root/"data"),"white_hot_samples":validate_white_hot(root/"data")},indent=2,ensure_ascii=False))
