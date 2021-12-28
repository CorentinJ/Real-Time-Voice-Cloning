import argparse
from pathlib import Path

from utils.argutils import print_args
from vocoder.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the vocoder from the synthesizer audios and the GTA synthesized mels, "
                    "or ground truth mels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("datasets_root", type=Path, help= \
        "Path to the directory containing your SV2TTS directory. Specifying --syn_dir or --voc_dir "
        "will take priority over this argument.")
    parser.add_argument("--syn_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("--voc_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the vocoder directory that contains the GTA synthesized mel spectrograms. "
        "Defaults to <datasets_root>/SV2TTS/vocoder/. Unused if --ground_truth is passed.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "Path to the directory that will contain the saved model weights, as well as backups "
        "of those weights and wavs generated during training.")
    parser.add_argument("-g", "--ground_truth", action="store_true", help= \
        "Train on ground truth spectrograms (<datasets_root>/SV2TTS/synthesizer/mels).")
    parser.add_argument("-s", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "syn_dir"):
        args.syn_dir = args.datasets_root / "SV2TTS" / "synthesizer"
    if not hasattr(args, "voc_dir"):
        args.voc_dir = args.datasets_root / "SV2TTS" / "vocoder"
    del args.datasets_root
    args.models_dir.mkdir(exist_ok=True)

    # Run the training
    print_args(args, parser)
    train(**vars(args))
