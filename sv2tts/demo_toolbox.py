from pathlib import Path
from toolbox import Toolbox

if __name__ == '__main__':
    datasets_root = Path(r"E:\Datasets")
    encoder_fpath = Path("encoder/saved_models/pretrained.pt")
    Toolbox(datasets_root, encoder_fpath)
    