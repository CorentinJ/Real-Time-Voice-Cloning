from pathlib import Path
from toolbox import Toolbox

if __name__ == '__main__':
    datasets_root = Path(r"E:\Datasets")
    encoder_models_dir = Path("encoder/saved_models")
    synthesizer_models_dir = Path("synthesizer/saved_models")
    vocoder_models_dir = Path("vocoder/saved_models")
    Toolbox(datasets_root, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
    