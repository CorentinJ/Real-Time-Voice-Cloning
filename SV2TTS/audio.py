import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import sounddevice
from params import *

def load(fpath, sampling_rate):
    """
    Loads a single audio file as a raw waveform.
    
    :param fpath: path to the audio file
    :param sampling_rate: the sampling rate of the audio. The audio will be resampled if the rate 
    differs from the original audio. 
    :return: the waveform as a numpy array of floats
    """
    return librosa.load(fpath, sr=sampling_rate)

def wave_to_mel_filterbank(wave, sampling_rate):
    """
    Converts a raw waveform into a mel spectrogram. Global parameters determine the window length
    and the window step. 
    
    :param wave: the raw waveform as a numpy array of floats
    :param sampling_rate: the sampling rate of the waveform
    :return: the features as a numpy array of shape (length, mel_n)
    """
    frames = librosa.feature.melspectrogram(
        wave, 
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n
    )
    return frames.astype(np.float32).transpose()

def plot_wave(wave, sampling_rate):
    plt.plot(wave)
    plt.show()
    
def plot_mel_filterbank(frames, sampling_rate):
    librosa.display.specshow(
        librosa.power_to_db(frames.transpose(), ref=np.max),
        hop_length=int(sampling_rate * 0.01),
        y_axis='mel',
        x_axis='time',
        sr=sampling_rate
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    
def play_wave(wave, sampling_rate, blocking=False):
    sounddevice.stop()
    sounddevice.play(wave, sampling_rate, blocking=blocking)
    