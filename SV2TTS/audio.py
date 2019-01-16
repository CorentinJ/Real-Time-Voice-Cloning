import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import sounddevice
import webrtcvad
import struct
from scipy.ndimage.morphology import binary_dilation
from params_data import *

def load(fpath):
    """
    Loads a single audio file as a raw waveform.
    
    :param fpath: path to the audio file
    :param sampling_rate: the sampling rate of the audio. The audio will be resampled if the rate 
    differs from the original audio. 
    :return: the waveform as a numpy array of floats
    """
    return librosa.load(fpath, sr=sampling_rate)[0]

def wave_to_mel_filterbank(wave):
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
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).transpose()

def trim_long_silences(wave):
    """
    Ensures that silences (segments without voice) in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    
    :param wave: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wave length)
    """
    # import matplotlib.pyplot as plt
    
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wave = wave[:len(wave) - (len(wave) % samples_per_window)]
    # plt.subplot(611)
    # plt.plot(wave)
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wave), *(wave * 32767).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wave), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    # plt.subplot(612)
    # plt.plot(voice_flags)
        
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    # plt.subplot(613)
    # plt.plot(audio_mask)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    # plt.subplot(614)
    # plt.plot(audio_mask)
    
    # Trim away the long silences in the audio
    audio_mask = np.repeat(audio_mask, samples_per_window)
    # plt.subplot(615)
    # plt.plot(wave)
    # plt.plot(audio_mask * 10000)
    
    wave = wave[audio_mask == True]
    # plt.subplot(616)
    # plt.plot(wave)
    # play_wave(wave)
    # plt.show()
    
    return wave
    
def plot_wave(wave):
    plt.plot(wave)
    plt.show()
    
def plot_mel_filterbank(frames):
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
    
def play_wave(wave, blocking=False):
    sounddevice.stop()
    sounddevice.play(wave, sampling_rate, blocking=blocking)


# if __name__ == '__main__':
#     fpath = r"E:\Datasets\LibriSpeech\train-other-500\37\215\37-215-0005.flac"
#     wave = load(fpath)
#     trim_long_silences(wave)

    