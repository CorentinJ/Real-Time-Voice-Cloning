from synthesizer.datasets import audio
from encoder import inference
from vlibs import fileio
import numpy as np
import os

encoder_model_fpath = 'SV2TTS/encoder/saved_models/all.pt'

def build_from_path(hparams, input_dirs, mel_dir, embed_dir, wav_dir):
    """
    Preprocesses the speech dataset from a given input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
        - embed_dir: output directory of the utterance embeddings
        - wav_dir: output directory of the preprocessed speech audio dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """
    
    data = []
    inference.load_model(encoder_model_fpath, 'cuda')
    print("Preprocessing utterances:")
    for input_dir in input_dirs:
        for speaker_dir in fileio.listdir(input_dir, full_path=True):
            print("    " + speaker_dir)
            for book_dir in fileio.listdir(speaker_dir, full_path=True):
                text_fpaths = fileio.get_files(book_dir, '\.normalized\.txt')
                wav_fpaths = fileio.get_files(book_dir, '\.wav')
                assert len(text_fpaths) == len(wav_fpaths)
                
                for text_fpath, wav_fpath in zip(text_fpaths, wav_fpaths):
                    basename = os.path.splitext(fileio.leaf(wav_fpath))[0]
                    text = fileio.read_all_lines(text_fpath)[0].rstrip()
                    text = text.lower()
                    data.append(_process_utterance(mel_dir, embed_dir, wav_dir, basename,
                                                   wav_fpath, text, hparams))

    n_all_samples = len(data)
    data = [d for d in data if d is not None]
    n_remaining_samples = len(data)
    print("Processed %d samples, pruned %d (remaining: %d)" %
          (n_all_samples, n_all_samples - n_remaining_samples, n_remaining_samples))
    return data

def _process_utterance(mel_dir, embed_dir, wav_dir, basename, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair.

    This writes the mel scale spectogram to disk and return a tuple to write to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - embed_dir: the directory to write the embedding into
        - wav_dir: the directory to write the preprocessed wav into
        - basename: the source base filename to use in the spectogram filename
        - wav_path: the path to the audio waveform
        - text: text spoken in the audio
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text)
    """
    wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    if hparams.rescale:
        wav = (wav / np.abs(wav).max()) * hparams.rescaling_max

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    ### SV2TTS ###
    # Compute the embedding of the utterance
    embed = inference.embed_utterance(wav)
    ##############
    
    # Write the spectrogram, embed and audio to disk
    audio_filename = 'audio-{}.npy'.format(basename)
    mel_filename = 'mel-{}.npy'.format(basename)
    embed_filename = 'embed-{}.npy'.format(basename)
    np.save(os.path.join(wav_dir, audio_filename), wav, allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(embed_dir, embed_filename), embed, allow_pickle=False)
    
    # Return a tuple describing this training example
    return audio_filename, mel_filename, embed_filename, len(wav), mel_frames, text
