from synthesizer.datasets import preprocessor
from synthesizer.hparams import hparams
from vlibs import fileio
import argparse
import os


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
    print('Initializing preprocessing..')
    parser = argparse.ArgumentParser()
    
    # Root data directory that contains the LibriTTS directory
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--output', default='Synthesizer')
    
    # Name of the LibriTTS sets to use, separated by spaces 
    # (e.g. "--sets train-other-500 train-clean-360). Defaults to using all the clean training sets 
    # present in the LibriSpeech directory.
    parser.add_argument('--sets', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    
    modified_hp = hparams.parse(args.hparams)
    
    run_preprocess(args, modified_hp)

if __name__ == '__main__':
    main()
