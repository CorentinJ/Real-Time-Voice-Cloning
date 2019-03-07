import os
from vlibs import fileio
import numpy as np
from datasets import audio
import sys
sys.path.append('../encoder')
encoder_model_fpath = '../encoder/saved_models/all.pt'
from encoder import inference

import matplotlib.pyplot as plt
import sounddevice as sd


def build_from_path(hparams, input_dirs, mel_dir, embed_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

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
            for utterance_dir in fileio.listdir(speaker_dir, full_path=True):
                alignment_file = fileio.get_files(utterance_dir, '.alignment.txt')[0]
                for line in fileio.read_all_lines(alignment_file):
                    # Retrieve the audio filepath and its alignment data
                    basename, words, end_times = line.strip().split(' ')
                    words = words.replace('\"', '').split(',')
                    end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                    wav_path = fileio.join(utterance_dir, basename + '.flac')

                    # Split utterances on silences
                    wavs, texts = _clean_and_split_utterance(wav_path, words, end_times, hparams)
                    
                    # Process all parts of the utterance
                    for i, (wav, text) in enumerate(zip(wavs, texts)):
                        sub_basename = "%s_%02d" % (basename, i)
                        data.append(_process_utterance(mel_dir, embed_dir, wav_dir, sub_basename,
                                                       wav, text, hparams))

    n_all_samples = len(data)
    data = [d for d in data if d is not None]
    n_remaining_samples = len(data)
    print("Processed %d samples, pruned %d (remaining: %d)" %
          (n_all_samples, n_all_samples - n_remaining_samples, n_remaining_samples))
    return data

def _clean_and_split_utterance(wav_path, words, end_times, hparams):
    # Load and rescale the audio
    wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    
    # Find pauses in the sentence
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == '' and words[-1] == ''
    
    # Break the sentence on pauses that are too long
    mask = (words == '') & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]
    segment_times = [[end_times[s], start_times[e]] for s, e in zip(breaks[:-1], breaks[1:])]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [' '.join(words[s + 1:e]).replace('  ', ' ') for s, e in zip(breaks[:-1], breaks[1:])]
    
    return wavs, texts

def _process_utterance(mel_dir, embed_dir, wav_dir, basename, wav, text, hparams):
    """
    Preprocesses a single utterance wav/text pair.

    This writes the mel scale spectogram to disk and return a tuple to write to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - embed_dir: the directory to write the embedding into
        - wav_dir: the directory to write the preprocessed wav into
        - basename: the source base filename to use in the spectogram filename
        - wav: the audio waveform unprocessed
        - text: text spoken in the audio
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text)
    """   
    ### SV2TTS ###
    # Compute the embedding of the utterance
    embed = inference.embed_utterance(wav)
    ##############
    
    out = wav
    constant_values = 0.
    out_dtype = np.float32
    
    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    if hparams.use_lws:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))
        
        # Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))
        
        # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, pad, mode='reflect')
    
    assert len(out) >= mel_frames * audio.get_hop_size(hparams)
    
    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)
    
    # Write the spectrogram, embed and audio to disk
    audio_filename = 'audio-{}.npy'.format(basename)
    mel_filename = 'mel-{}.npy'.format(basename)
    embed_filename = 'embed-{}.npy'.format(basename)
    np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(embed_dir, embed_filename), embed, allow_pickle=False)
    
    # Return a tuple describing this training example
    return audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text
