from synthesizer.synthesize import run_synthesis
from synthesizer.hparams import hparams
from utils.argutils import print_args
import argparse
import os


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Creates ground-truth aligned (GTA) spectrograms from the vocoder.",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=str, help=\
        "Path to the directory containing your SV2TTS directory. If you specify both --in_dir and "
        "--out_dir, this argument won't be used.")
    parser.add_argument("--model_dir", type=str, 
                        default="synthesizer/saved_models/pretrained/", help=\
        "Path to the pretrained model directory.")
    parser.add_argument("-i", "--in_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the mel spectrograms, the wavs and the "
        "embeds. Defaults to  <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-o", "--out_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the output vocoder directory that will contain the ground truth aligned mel "
        "spectrograms. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value "
                             "pairs")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    args = parser.parse_args()
    print_args(args, parser)
    modified_hp = hparams.parse(args.hparams)
    
    if not hasattr(args, "in_dir"):
        args.in_dir = os.path.join(args.datasets_root, "SV2TTS", "synthesizer")
    if not hasattr(args, "out_dir"):
        args.out_dir = os.path.join(args.datasets_root, "SV2TTS", "vocoder")

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Verify webrtcvad is available
    if not args.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    run_synthesis(args.in_dir, args.out_dir, args.model_dir, modified_hp)

