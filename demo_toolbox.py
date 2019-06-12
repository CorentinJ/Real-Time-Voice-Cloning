from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.")
    parser.add_argument("-e", "--enc_models_dir", type=Path, default="encoder/saved_models", 
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default="synthesizer/saved_models", 
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default="vocoder/saved_models", 
                        help="Directory containing saved vocoder models")
    args = parser.parse_args()

    # Launch the toolbox
    print_args(args, parser)
    Toolbox(**vars(args))
    