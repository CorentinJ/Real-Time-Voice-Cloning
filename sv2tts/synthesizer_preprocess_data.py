from synthesizer.datasets.preprocessor import preprocess_librispeech
from synthesizer.hparams import hparams
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=str, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-m", "--out_dir", type=str, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms. Defaults to "
        "<datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-w", "--wav_out_dir", type=str, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the audio wav files to train the"
        "vocoder. Defaults to <datasets_root>/SV2TTS/vocoder/ if out_dir is not set either."
        "If out_dir is set but wav_out_dir is not, wav_out_dir will default to out_dir.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    args = parser.parse_args()
    args.hparams = hparams.parse(args.hparams)
    
    # Process the arguments
    args.datasets_root = Path(args.datasets_root)
    if not hasattr(args, "wav_out_dir"):
        if not hasattr(args, "out_dir"):
            args.wav_out_dir = Path(args.datasets_root, "SV2TTS", "vocoder")
        else:
            args.wav_out_dir = args.out_dir
    if not hasattr(args, "out_dir"):
        args.out_dir = Path(args.datasets_root, "SV2TTS", "synthesizer")
    args.out_dir = Path(args.out_dir)
    args.wav_out_dir = Path(args.wav_out_dir)

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    args.wav_out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    preprocess_librispeech(**vars(args))    

if __name__ == "__main__":
    main()
