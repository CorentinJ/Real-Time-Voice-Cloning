"""
Demo Toolbox
============

This module contains the commands and parser arguments necessary
to run the Toolbox in the intended way. The Toolbox is a GUI simplifying
the interaction with the code's functionality.
"""

from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
from utils.modelutils import check_model_paths
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the Toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pass optional arguments to the parser
    parser.add_argument(
        "-d",
        "--datasets_root",
        type=Path,
        help="Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.",
        default=None,
    )

    parser.add_argument(
        "-e",
        "--enc_models_dir",
        type=Path,
        default="encoder/saved_models",
        help="Directory containing saved encoder models",
    )

    parser.add_argument(
        "-s",
        "--syn_models_dir",
        type=Path,
        default="synthesizer/saved_models",
        help="Directory containing saved synthesizer models",
    )

    parser.add_argument(
        "-v",
        "--voc_models_dir",
        type=Path,
        default="vocoder/saved_models",
        help="Directory containing saved vocoder models",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="If True, processing is done on CPU, even if a GPU is available.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random number seed value to make toolbox deterministic.",
    )

    parser.add_argument(
        "--no_mp3_support",
        action="store_true",
        help="If True, mp3 files are not allowed.",
    )

    # Parse given arguments
    args = parser.parse_args()
    print_args(args, parser)

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    del args.cpu

    ## Remind user to download pretrained models if needed
    check_model_paths(
        encoder_path=args.enc_models_dir,
        synthesizer_path=args.syn_models_dir,
        vocoder_path=args.voc_models_dir,
    )

    # Launch the toolbox
    Toolbox(**vars(args))
