from synthesizer.datasets import preprocessor
from synthesizer.hparams import hparams
from pathlib import Path
import argparse


def preprocess(args, input_folders, out_dir, hparams):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    embed_dir = os.path.join(out_dir, 'embed')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, embed_dir, wav_dir)
    write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def norm_data(args):
    print('Selecting data folders..')
    dataset_dir = fileio.join(args.base_dir, 'LibriTTS')
    if args.sets is not None:
        sets = args.sets
    else:
        sets = [set for set in fileio.listdir(dataset_dir) if set.startswith('train-clean')]
    return fileio.join(dataset_dir, sets)

def run_preprocess(args, hparams):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)
    preprocess(args, input_folders, output_folder, hparams)

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms, "
                    "retrieves their speaker embedding with the speaker encoder and writes them to "
                    "the disk. Audio files are also saved, to be used by the vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Root data directory that contains the LibriTTS directory
    parser.add_argument("datasets_root", type=str, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-e", "--encoder_fpath", type=str, 
                        default='encoder/saved_models/pretrained.pt', help=\
        "Path your trained encoder model.")
    parser.add_argument("-m", "--mel_out_dir", type=str, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms and the speaker "
        "embeddings. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-w", "--wav_out_dir", type=str, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the audio wav files to train the"
        "vocoder. Defaults to <datasets_root>/SV2TTS/vocoder/")
    parser.add_argument("-h", "--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    
    args = parser.parse_args()
    modified_hp = hparams.parse(args.hparams)
    
    run_preprocess(args, modified_hp)

if __name__ == '__main__':
    main()
