from params_data import *
from params_model import model_embedding_size
from preprocess import preprocess_wave
from config import model_dir, device
from model import SpeakerEncoder
from vlibs import fileio
import numpy as np
import audio
import torch


default_weights_fpath = fileio.join(model_dir, 'all.pt') 
_model = None # type: SpeakerEncoder

def load_model(weights_fpath=default_weights_fpath):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    """
    global _model
    _model = SpeakerEncoder()
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

def embed_frames_batch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram.
    
    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        load_model()
    
    frames = torch.from_numpy(frames_batch).to(device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed

def compute_partial_splits(n_samples, partial_utterance_n_frames=partial_utterance_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes 
    
    :param n_samples: 
    :param partial_utterance_n_frames: 
    :param min_pad_coverage: 
    :param overlap: 
    :return: 
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round(partial_utterance_n_frames * (1 - overlap)))

    # Compute the splits
    wave_splits, mel_splits = [], []
    for i in range(0, n_frames - partial_utterance_n_frames + frame_step + 1, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wave_range = mel_range * samples_per_frame
        mel_splits.append(slice(*mel_range))
        wave_splits.append(slice(*wave_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wave_range = wave_splits[-1]
    coverage = (n_samples - last_wave_range.start) / (last_wave_range.stop - last_wave_range.start)
    if coverage < min_pad_coverage:
        mel_splits = mel_splits[:-1]
        wave_splits = wave_splits[:-1]
    
    return wave_splits, mel_splits

def embed_utterance(wave, using_partials=True, independant_partials=True,
                    return_partial_embeds=False, return_partial_waves=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    :param wave: the utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param independant_partials: if True, partial utterances will TODO: is this even interesting?
    :param return_partial_embeds: if True, the partial embeddings will also be returned. Requires 
    <using_partials> to be True.
    :param return_partial_waves: if True, the wave segments corresponding to the partial embeds 
    will also be returned. Requires <using_partials> to be True.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,), 
    the partial utterances as a numpy array of float32 of shape (n_partials, model_embedding_size,)
    [optional] and the wave partials as a numpy array of float32 of shape (n_partials, n_samples)
    [optional].
    """
    if not using_partials and (return_partial_embeds or independant_partials or return_partial_waves):
        raise ValueError("Cannot use partial utterances when complete utterance mode is set.")
    
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wave_to_mel_filterbank(wave)
        return embed_frames_batch(frames[None, ...])[0]
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_splits, mel_splits = compute_partial_splits(len(wave), **kwargs)
    max_wave_length = wave_splits[-1].stop
    if max_wave_length >= len(wave):
        wave = np.pad(wave, (0, max_wave_length - len(wave)), 'constant')
    
    # Split the utterance into partials
    frames = audio.wave_to_mel_filterbank(wave)
    frames_batch = np.array([frames[s] for s in mel_splits])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    out = [embed]
    
    if return_partial_embeds:
        out.append(partial_embeds)
    
    if return_partial_waves:
        out.append(np.array([wave[s] for s in wave_splits]))
        
    return tuple(out)
    
def embed_stream(stream, partial_utterance_n_frames=partial_utterance_n_frames, overlap=0.5):
    pass

def embed_speaker(waves, normalize=False, **kwargs):
    pass

def load_and_preprocess_wave(fpath):
    """
    Loads an audio file in memory and applies the same preprocessing operations used in trained 
    the Speaker Encoder. Using this function is not mandatory but recommended.
    
    :param fpath: the path to an audio file. Several extensions are supported (mp3, wav, flac, ...)
    :return: the audio waveform as a numpy array of float32.
    """
    wave = audio.load(fpath)
    wave = preprocess_wave(wave)
    return wave

if __name__ == '__main__':
    from time import perf_counter
    
    fpath = r"E:\Datasets\LibriSpeech\train-other-500\149\125760\149-125760-0003.flac"
    wave = load_and_preprocess_wave(fpath)
    
    start = perf_counter()
    load_model()
    print("Loaded model in %.2fs" % (perf_counter() - start))

    duration = len(wave) / sampling_rate
    start = perf_counter()
    embed = embed_utterance(wave)
    print("Processed %.2fs long utterance in %.2fs" % (duration, perf_counter() - start))
    
    print(embed)
    