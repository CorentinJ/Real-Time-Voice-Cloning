import torch
torch.manual_seed(0)

print('Running in 1-thread CPU mode for fully reproducible results training a CNN and generating numpy randomness.  This mode may be slow...')
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
seed_value += 1

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
seed_value += 1

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
seed_value += 1

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
from utils.modelutils import check_model_paths
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets. You can add your own data by created a directory named UserAudio "
        "in your datasets root. Supported formats are mp3, flac, wav and m4a. Each speaker should "
        "be inside a directory, e.g. <datasets_root>/UserAudio/speaker_01/audio_01.wav.",
                        default=None)
    parser.add_argument("-e", "--enc_models_dir", type=Path, default="encoder/saved_models", 
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default="synthesizer/saved_models", 
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default="vocoder/saved_models", 
                        help="Directory containing saved vocoder models")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    args = parser.parse_args()
    print_args(args, parser)

    ## Remind the user to download pretrained models if needed
    check_model_paths(encoder_path=args.enc_models_dir, synthesizer_path=args.syn_models_dir,
                      vocoder_path=args.voc_models_dir)

    # Launch the toolbox
    Toolbox(**vars(args))    
