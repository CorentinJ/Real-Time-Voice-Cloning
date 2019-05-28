from vocoder.train import train
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the vocoder from the synthesizer audios and the GTA synthesized mels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("datasets_root", type=str, help= \
        "Path to the directory containing your SV2TTS directory. If you specify both --syn_dir "
        "and --voc_dir, this argument won't be used.")
    parser.add_argument("--syn_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the mel spectrograms, the wavs and the "
        "embeds. Defaults to  <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("--voc_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the vocoder directory that contains the ground truth aligned mel "
        "spectrograms. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    parser.add_argument("-m", "--models_dir", type=str, default="vocoder/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    # parser.add_argument("-s", "--save_every", type=int, default=500, help= \
    #     "Number of steps between updates of the model on the disk. Set to 0 to never save the "
    #     "model.")
    # parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
    #     "Number of steps between backups of the model. Set to 0 to never make backups of the "
    #     "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model.")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "syn_dir"):
        args.syn_dir = Path(args.datasets_root, "SV2TTS", "synthesizer")
    args.syn_dir = Path(args.syn_dir)
    if not hasattr(args, "voc_dir"):
        args.voc_dir = Path(args.datasets_root, "SV2TTS", "vocoder")
    args.voc_dir = Path(args.voc_dir)
    del args.datasets_root
    args.models_dir = Path(args.models_dir)
    args.models_dir.mkdir(exist_ok=True)

    # Run the training
    train(**vars(args))
    