import numpy as np
import torch
from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder import audio

_model = None # type: SpeakerEncoder
_device = None # type: torch.device

def load_model(weights_fpath, device):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. 'cpu', 'cuda'). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu.
    """
    global _model, _device
    if isinstance(device, str):
        device = torch.device(device)
    _device = device
    _model = SpeakerEncoder(_device)
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
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed

def compute_partial_splits(n_samples, partial_utterance_n_frames=partial_utterance_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram splits are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_splits[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 split.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform splits and mel spectrogram splits as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the splits
    wave_splits, mel_splits = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wave_range = mel_range * samples_per_frame
        mel_splits.append(slice(*mel_range))
        wave_splits.append(slice(*wave_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wave_range = wave_splits[-1]
    coverage = (n_samples - last_wave_range.start) / (last_wave_range.stop - last_wave_range.start)
    if coverage < min_pad_coverage and len(mel_splits) > 1:
        mel_splits = mel_splits[:-1]
        wave_splits = wave_splits[:-1]
    
    return wave_splits, mel_splits

def embed_utterance(wave, using_partials=True, return_partial_embeds=False, 
                    return_wave_splits=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    :param wave: the utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param return_partial_embeds: if True, the partial embeddings will also be returned. Requires 
    <using_partials> to be True.
    :param return_wave_splits: if True, the wave split ranges corresponding to the partial embeds 
    will also be returned. Requires <using_partials> to be True.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,), 
    the partial utterances as a numpy array of float32 of shape (n_partials, model_embedding_size,)
    [optional] and the wave partials as a list of slices [optional].
    """
    if not using_partials and (return_partial_embeds or return_wave_splits):
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
    if return_wave_splits:
        out.append(wave_splits)
    return out[0] if len(out) == 1 else tuple(out)
    
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
    wave = audio.preprocess_wave(wave)
    return wave
