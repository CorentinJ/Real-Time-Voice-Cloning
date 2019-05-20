from pathlib import Path
from toolbox import Toolbox

if __name__ == '__main__':
    datasets_root = Path(r"E:\Datasets")
    Toolbox(datasets_root, "encoder/saved_models/pretrained.pt")
    