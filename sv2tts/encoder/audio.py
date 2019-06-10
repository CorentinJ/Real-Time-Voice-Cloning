from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import webrtcvad
import struct
from encoder.params_data import *


int16_max = (2 ** 15) - 1

def load(fpath):
    return librosa.load(fpath, sr=sampling_rate)[0]

def wav_to_mel_filterbank(wav):
    frames = librosa.feature.melspectrogram(
        wav, 
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).transpose()

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    
    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # import matplotlib.pyplot as plt
    
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    # _, axs = plt.subplots(3, 1, sharex=True)
    # for ax in axs:
    #     ax.plot((wav / np.max(np.abs(wav)) + 1) / 2)

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    # axs[0].set_title("Raw VAD")
    # axs[0].plot(np.repeat(voice_flags, samples_per_window))
        
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    # axs[1].set_title("Moving average + Binarization")
    # axs[1].plot(np.repeat(audio_mask, samples_per_window))
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    # axs[2].set_title("Binary dilation")
    # axs[2].plot(audio_mask)
    # for ax in axs:
    #     ax.set_yticks([])
    # xticks = np.arange(0, len(wav), sampling_rate)
    # plt.xticks(xticks, ["%ds" % (xi // sampling_rate) for xi in xticks])
    # plt.show()
    # quit()
    
    # Trim away the long silences in the audio
    return wav[audio_mask == True]
  
def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

def preprocess_wav(wav):
    """ 
    This is the standard routine that should be used on every audio file before being used in 
    this project.
    """
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    return wav

def plot_wave(wav):
    plt.plot(wav)
    plt.show()
    
def plot_mel_filterbank(frames):
    librosa.display.specshow(
        librosa.power_to_db(frames.transpose(), ref=np.max),
        hop_length=int(sampling_rate * 0.01),
        y_axis="mel",
        x_axis="time",
        sr=sampling_rate
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram")
    plt.tight_layout()
    plt.show()
    