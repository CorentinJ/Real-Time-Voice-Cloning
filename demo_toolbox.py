import argparse
import os
from pathlib import Path

from toolbox import Toolbox
from utils.argutils import print_args
from utils.default_models import ensure_default_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.", default=None)
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models",
                        help="Directory containing all saved models")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, all inference will be done on CPU")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Remind the user to download pretrained models if needed
    ensure_default_models(args.models_dir)

    # Launch the toolbox
    Toolbox(**arg_dict)
