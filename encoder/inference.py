from tokenize import Double
from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav   # We want to expose this function from here
from matplotlib import cm
from encoder import audio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

#debugging
import torchsnooper

## Added for AutoVC
from librosa.filters import mel

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from autovc.utils import butter_highpass, pySTFT


_model = None # type: SpeakerEncoder
_device = None # type: torch.device


def load_model(weights_fpath: Path, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))
    
    
def is_loaded():
    return _model is not None


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


def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices

def process_autovc(wav):
    device = torch.device(torch.cuda.current_device())
    
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))

    b, a = butter_highpass(30, 16000, order=5)
    # Unlike in Speechsplit we do not apply an Infinite Impulse Response (IIR) filter
    # TODO: Might need to change that?
    
    # Create Mel from wav
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0,1)

    # Extract F0 (Frequency)
    
    # Might be able to use mel generated in embed_utterances (audio.wav_to_mel_spectogram)
    melsp = S
    
    # len_crop = 256
    # left = np.random.randint(0, tmp.shape[0]-len_crop)
    # melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        
    
    return melsp

def embed_utterance_old(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_utterance(wav, G, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: REMOVED
    :param return_partials: REMOVED
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """

    # !REMOVED PARTIALS DUE TO INCOMPATIBILITY WITH AUTOVC
    # Process the entire utterance if not using partials
    # if not using_partials:
    #     frames = audio.wav_to_mel_spectrogram(wav)
    #     embed = embed_frames_batch(frames[None, ...])[0]
    #     if return_partials:
    #         return embed, None, None
    #     return embed
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Generate SpeechSplit on whole wav file
    # TODO: Generate per splitted utterance?
    mel_real_pad = process_autovc(wav)

    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed_orig = raw_embed / np.linalg.norm(raw_embed, 2)

    # Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    

    # Crop Spectogram to len crop because
    # Output of autovc encoder depends on length of mel

    #Todo either fix or get len crop from hparams
    tmp = mel_real_pad
    len_crop = 512
    if tmp.shape[0] <  len_crop:
        len_pad =  len_crop - tmp.shape[0]
        uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
    elif tmp.shape[0] >  len_crop:
        left = np.random.randint(tmp.shape[0]- len_crop)
        uttr = tmp[left:left+ len_crop, :]
    else:
        uttr = tmp
    
    
    # crop = int((mel_real_pad.shape[0] / 16)) * 16
    mel_real_pad = uttr
    embed_orig = torch.from_numpy(np.stack(embed_orig, axis=0))
    mel_real_pad = torch.from_numpy(np.stack(mel_real_pad, axis=0))

    mel_real_pad = torch.unsqueeze( mel_real_pad.to(device), 0)
    embed_orig = torch.unsqueeze(embed_orig.to(device), 0)
    
    embed = G(mel_real_pad.type(torch.cuda.FloatTensor), embed_orig, None)
    
    embed = embed.cpu().detach().numpy().flatten()
    embed = np.concatenate((embed_orig.cpu().detach().numpy().flatten(), embed))
    
    # !REMOVED PARTIALS DUE TO INCOMPATIBILITY WITH AUTOVC
    # if return_partials:
    #     return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplemented()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        # pad if shape is not devisible by height
        if(len(embed) != height**2):
            height += 1
            embed = np.pad(embed, (0, (height**2 - len(embed))), 'constant', constant_values=(0,0))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)
    
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
