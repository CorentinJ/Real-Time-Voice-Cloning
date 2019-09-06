import struct
import librosa
import numpy as np
import webrtcvad
from encoder.params_data import *
from scipy.ndimage import binary_dilation

from encoder.params_data import int16_max


def mean_dBFS(wav: np.ndarray) -> np.float64:
    """
    dBFS of wav

    20 * log10(sqrt(np.mean((wav * max_possible_amplitude)^2)) / max_possible_amplitude)
     = 20 * log10(sqrt(np.mean(wav^2)))
     = 10 * log10(np.mean(wav^2))
    :param wav: numpy array contains normalized ([-1.0, 1.0]) 16-bit mono audio samples
    :return:
    """
    dBFS = 10 * np.log10(np.mean(np.square(wav)))
    return dBFS


def max_dBFS(wav: np.ndarray) -> np.float64:
    """
    max dBFS of wav

    20 * log10(max(abs(sample)) * max_possible_amplitude / max_possible_amplitude) = 20 * log10(max(abs(sample)))

    :param wav: numpy array contains normalized ([-1.0, 1.0]) 16-bit mono audio samples
    :return:
    """
    dBFS = 20 * np.log10(np.max(np.abs(wav)))
    return dBFS


def trim_long_silences(wav: np.ndarray, vad: webrtcvad.Vad = None) -> np.ndarray:
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :param vad: an webrtcvad.Vad object. A new one with mode=3 will be created if None.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack(f'{len(wav)}h', *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    if vad is None:
        vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask]


def wav_to_mel_spectrogram(wav: np.ndarray):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def normalize_volume(wav: np.ndarray, target_dBFS: float, increase_only=False, decrease_only=False) -> np.ndarray:
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")

    wav_dBFS = mean_dBFS(wav)
    dBFS_change = target_dBFS - wav_dBFS
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def preprocess_wav(wav: np.ndarray,
                   source_sr: int) -> np.ndarray:
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param wav: the waveform as a numpy array of floats.
    :param source_sr: the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters.
    """
    # Resample the wav if needed
    if source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    # FIXME! VAD result could change significantly after normalization,
    #  a background audio is not filtered at all
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    wav = trim_long_silences(wav)

    return wav
