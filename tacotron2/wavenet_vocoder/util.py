import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa.display as dsp
import numpy as np
import tensorflow as tf


def _assert_valid_input_type(s):
	assert s == 'mulaw-quantize' or s == 'mulaw' or s == 'raw'

def is_mulaw_quantize(s):
	_assert_valid_input_type(s)
	return s == 'mulaw-quantize'

def is_mulaw(s):
	_assert_valid_input_type(s)
	return s == 'mulaw'

def is_raw(s):
	_assert_valid_input_type(s)
	return s == 'raw'

def is_scalar_input(s):
	return is_raw(s) or is_mulaw(s)


#From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def mulaw(x, mu=256):
	"""Mu-Law companding
	Method described in paper [1]_.
	.. math::
		f(x) = sign(x) ln (1 + mu |x|) / ln (1 + mu)
	Args:
		x (array-like): Input signal. Each value of input signal must be in
		  range of [-1, 1].
		mu (number): Compression parameter ``μ``.
	Returns:
		array-like: Compressed signal ([-1, 1])
	See also:
		:func:`nnmnkwii.preprocessing.inv_mulaw`
		:func:`nnmnkwii.preprocessing.mulaw_quantize`
		:func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
	.. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
		implementations using the tms320c54x." SPRA163 (1997).
	"""
	mu = 255
	return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
	"""Inverse of mu-law companding (mu-law expansion)
	.. math::
		f^{-1}(x) = sign(y) (1 / mu) (1 + mu)^{|y|} - 1)
	Args:
		y (array-like): Compressed signal. Each value of input signal must be in
		  range of [-1, 1].
		mu (number): Compression parameter ``μ``.
	Returns:
		array-like: Uncomprresed signal (-1 <= x <= 1)
	See also:
		:func:`nnmnkwii.preprocessing.inv_mulaw`
		:func:`nnmnkwii.preprocessing.mulaw_quantize`
		:func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
	"""
	mu = 255
	return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
	"""Mu-Law companding + quantize
	Args:
		x (array-like): Input signal. Each value of input signal must be in
		  range of [-1, 1].
		mu (number): Compression parameter ``μ``.
	Returns:
		array-like: Quantized signal (dtype=int)
		  - y ∈ [0, mu] if x ∈ [-1, 1]
		  - y ∈ [0, mu) if x ∈ [-1, 1)
	.. note::
		If you want to get quantized values of range [0, mu) (not [0, mu]),
		then you need to provide input signal of range [-1, 1).
	Examples:
		>>> from scipy.io import wavfile
		>>> import pysptk
		>>> import numpy as np
		>>> from nnmnkwii import preprocessing as P
		>>> fs, x = wavfile.read(pysptk.util.example_audio_file())
		>>> x = (x / 32768.0).astype(np.float32)
		>>> y = P.mulaw_quantize(x)
		>>> print(y.min(), y.max(), y.dtype)
		15 246 int64
	See also:
		:func:`nnmnkwii.preprocessing.mulaw`
		:func:`nnmnkwii.preprocessing.inv_mulaw`
		:func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
	"""
	mu = 255
	y = mulaw(x, mu)
	# scale [-1, 1] to [0, mu]
	return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
	"""Inverse of mu-law companding + quantize
	Args:
		y (array-like): Quantized signal (∈ [0, mu]).
		mu (number): Compression parameter ``μ``.
	Returns:
		array-like: Uncompressed signal ([-1, 1])
	Examples:
		>>> from scipy.io import wavfile
		>>> import pysptk
		>>> import numpy as np
		>>> from nnmnkwii import preprocessing as P
		>>> fs, x = wavfile.read(pysptk.util.example_audio_file())
		>>> x = (x / 32768.0).astype(np.float32)
		>>> x_hat = P.inv_mulaw_quantize(P.mulaw_quantize(x))
		>>> x_hat = (x_hat * 32768).astype(np.int16)
	See also:
		:func:`nnmnkwii.preprocessing.mulaw`
		:func:`nnmnkwii.preprocessing.inv_mulaw`
		:func:`nnmnkwii.preprocessing.mulaw_quantize`
	"""
	# [0, m) to [-1, 1]
	mu = 255
	y = 2 * _asfloat(y) / mu - 1
	return inv_mulaw(y, mu)

def _sign(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.sign(x) if (isnumpy or isscalar) else tf.sign(x)


def _log1p(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.log1p(x) if (isnumpy or isscalar) else tf.log1p(x)


def _abs(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.abs(x) if (isnumpy or isscalar) else tf.abs(x)


def _asint(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return x.astype(np.int) if isnumpy else int(x) if isscalar else tf.cast(x, tf.int32)


def _asfloat(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return x.astype(np.float32) if isnumpy else float(x) if isscalar else tf.cast(x, tf.float32)

def sequence_mask(input_lengths, max_len=None, expand=True):
	if max_len is None:
		max_len = tf.reduce_max(input_lengths)

	if expand:
		return tf.expand_dims(tf.sequence_mask(input_lengths, max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(input_lengths, max_len, dtype=tf.float32)


def waveplot(path, y_hat, y_target, hparams):
	sr = hparams.sample_rate

	plt.figure(figsize=(12, 4))
	if y_target is not None:
		ax = plt.subplot(2, 1, 1)
		dsp.waveplot(y_target, sr=sr)
		ax.set_title('Target waveform')
		ax = plt.subplot(2, 1, 2)
		dsp.waveplot(y_hat, sr=sr)
		ax.set_title('Prediction waveform')
	else:
		ax = plt.subplot(1, 1, 1)
		dsp.waveplot(y_hat, sr=sr)
		ax.set_title('Generated waveform')

	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()
