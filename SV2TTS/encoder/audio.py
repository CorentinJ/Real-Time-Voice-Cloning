from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import sounddevice
import webrtcvad
import struct
from encoder.params_data import *

int16_max = (2 ** 15) - 1

def load(fpath):
    return librosa.load(fpath, sr=sampling_rate)[0]

def wave_to_mel_filterbank(wave):
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
    Ensures that segments without voice in the waveform remain no longer than a 
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
    pcm_wave = struct.pack("%dh" % len(wave), *(np.round(wave * int16_max)).astype(np.int16))
    
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
  
def normalize_volume(wave, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wave * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wave
    return wave * (10 ** (dBFS_change / 20))

def preprocess_wave(wave):
    """ 
    This is the standard routine that should be used on every audio file before being used in 
    this project.
    """
    wave = normalize_volume(wave, audio_norm_target_dBFS, increase_only=True)
    wave = trim_long_silences(wave)
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
    
def rec_wave(duration, blocking=True, verbose=True):
    if verbose:
        print('Recording %d seconds of audio' % duration)
    wave = sounddevice.rec(duration * sampling_rate, sampling_rate, 1)
    if blocking:
        sounddevice.wait()
        if verbose:
            print('Done recording!')
    return wave.squeeze()
