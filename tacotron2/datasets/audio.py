import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import webrtcvad
import struct
from scipy.ndimage.morphology import binary_dilation


def load_wav(path, sr):
	return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
	librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
	if preemphasize:
		return signal.lfilter([1, -k], [1], wav)
	return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
	if inv_preemphasize:
		return signal.lfilter([1], [1, -k], wav)
	return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
	for start in range(quantized.size):
		if abs(quantized[start] - 127) > silence_threshold:
			break
	for end in range(quantized.size - 1, 1, -1):
		if abs(quantized[end] - 127) > silence_threshold:
			break

	assert abs(quantized[start] - 127) > silence_threshold
	assert abs(quantized[end] - 127) > silence_threshold

	return start, end

# def trim_silence(wav, hparams):
# 	'''Trim leading and trailing silence
# 
# 	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
# 	'''
# 	# Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. 
#     # These params are separate and tunable per dataset.
# 	return librosa.effects.trim(wav, top_db=hparams.trim_top_db, 
# 								frame_length=hparams.trim_fft_size, 
# 								hop_length=hparams.trim_hop_size)[0]

def trim_silence(wav, hparams):
	"""
    Ensures that silences (segments without voice) in the waveform remain no longer than a 
    threshold determined by the VAD parameters in hparams.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wave length)
    """
	
	# Compute the voice detection window size
	samples_per_window = (hparams.vad_window_length * hparams.sample_rate) // 1000
	
	# Trim the end of the audio to have a multiple of the window size
	wav = wav[:len(wav) - (len(wav) % samples_per_window)]
	
	# Convert the float waveform to 16-bit mono PCM
	pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * 32767)).astype(np.int16))
	
	# Perform voice activation detection
	voice_flags = []
	vad = webrtcvad.Vad(mode=3)
	for window_start in range(0, len(wav), samples_per_window):
		window_end = window_start + samples_per_window
		voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
										 sample_rate=hparams.sample_rate))
	voice_flags = np.array(voice_flags)
	
	# Smooth the voice detection with a moving average
	def moving_average(array, width):
		array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
		ret = np.cumsum(array_padded, dtype=float)
		ret[width:] = ret[width:] - ret[:-width]
		return ret[width - 1:] / width
	
	audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
	audio_mask = np.round(audio_mask).astype(np.bool)
	
	# Dilate the voiced regions
	audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
	
	# Trim away the long silences in the audio
	audio_mask = np.repeat(audio_mask, samples_per_window)
	wave = wav[audio_mask == True]
	
	return wave

def get_hop_size(hparams):
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size

def linearspectrogram(wav, hparams):
	D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
	S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def melspectrogram(wav, hparams):
	D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
	S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
	'''Converts linear spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(linear_spectrogram, hparams)
	else:
		D = linear_spectrogram

	S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
	else:
		return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, hparams):
	'''Converts mel spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(mel_spectrogram, hparams)
	else:
		D = mel_spectrogram

	S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
	else:
		return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def _lws_processor(hparams):
	import lws
	return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles, hparams)
	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y, hparams)))
		y = _istft(S_complex * angles, hparams)
	return y

def _stft(y, hparams):
	if hparams.use_lws:
		return _lws_processor(hparams).stft(y).T
	else:
		return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y, hparams):
	return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
	"""Compute number of time frames of spectrogram
	"""
	pad = (fsize - fshift)
	if length % fshift == 0:
		M = (length + pad * 2 - fsize) // fshift + 1
	else:
		M = (length + pad * 2 - fsize) // fshift + 2
	return M


def pad_lr(x, fsize, fshift):
	"""Compute left and right padding
	"""
	M = num_frames(len(x), fsize, fshift)
	pad = (fsize - fshift)
	T = len(x) + 2 * pad
	r = (M - 1) * fshift + fsize - T
	return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
	'''compute right padding (final frame)
	'''
	return int(fsize // 2)


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis(hparams)
	return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
		else:
			return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

	assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
	if hparams.symmetric_mels:
		return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
	else:
		return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return (((np.clip(D, -hparams.max_abs_value,
				hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
				+ hparams.min_level_db)
		else:
			return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

	if hparams.symmetric_mels:
		return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
	else:
		return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
