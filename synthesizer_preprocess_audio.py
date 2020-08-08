from synthesizer_pt.preprocess import preprocess_dataset
from synthesizer_pt import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse
from typing import NamedTuple


class hyperparameters(NamedTuple):
    # This is a workaround because multiprocessing cannot pickle the "hparams" module
    fmin: float
    hop_length: int
    max_abs_value: float
    max_mel_frames: int
    min_level_db: float
    num_mels: int
    n_fft: int
    preemphasis: float
    preemphasize: bool
    ref_level_db: float
    rescale: bool
    rescaling_max: float
    sample_rate: int
    signal_normalization: bool
    silence_min_duration_split: float
    utterance_min_duration: float
    win_length: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=None, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    parser.add_argument("--no_alignments", action="store_true", help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("--datasets_name", type=str, default="LibriSpeech", help=\
        "Name of the dataset directory to process.")
    parser.add_argument("--subfolders", type=str, default="train-clean-100, train-clean-360", help=\
        "Comma-separated list of subfolders to process inside your dataset directory")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Verify webrtcvad is available
    if not args.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    # Preprocess the dataset
    print_args(args, parser)

    # Pass the hparams to preprocess as a NamedTuple instead of a module to allow multiprocessing
    args.hparams = hyperparameters(sample_rate = hparams.sample_rate,
                                   rescale = hparams.tts_rescale,
                                   rescaling_max = hparams.tts_rescaling_max,
                                   utterance_min_duration = hparams.tts_utterance_min_duration,
                                   preemphasis = hparams.preemphasis,
                                   preemphasize = hparams.preemphasize,
                                   n_fft = hparams.n_fft,
                                   hop_length = hparams.hop_length,
                                   win_length = hparams.win_length,
                                   num_mels = hparams.num_mels,
                                   fmin = hparams.fmin,
                                   min_level_db = hparams.min_level_db,
                                   ref_level_db = hparams.ref_level_db,
                                   signal_normalization = hparams.signal_normalization,
                                   max_abs_value = hparams.max_abs_value,
                                   max_mel_frames = hparams.tts_max_mel_len,
                                   silence_min_duration_split = hparams.tts_silence_min_duration_split,
                                   #n_fft = hparams.n_fft,
                                   #n_fft = hparams.n_fft,
                                   #n_fft = hparams.n_fft,
                                  )
    preprocess_dataset(**vars(args))
